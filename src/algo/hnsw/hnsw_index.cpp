#include "algo/hnsw/hnsw_index.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/index/index_type_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/setting_info.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "algo/hnsw/hnsw_module.hpp"
#include "vindex/vector_index_registry.hpp"

#include <cstring>

namespace duckdb {
namespace vindex {
namespace hnsw {

//------------------------------------------------------------------------------
// Options parsing
//------------------------------------------------------------------------------

namespace {

MetricKind ParseMetric(const case_insensitive_map_t<Value> &options) {
	auto it = options.find("metric");
	if (it == options.end()) {
		return MetricKind::L2SQ;
	}
	const auto raw = StringUtil::Lower(it->second.GetValue<string>());
	if (raw == "l2sq") {
		return MetricKind::L2SQ;
	}
	if (raw == "cosine") {
		return MetricKind::COSINE;
	}
	if (raw == "ip") {
		return MetricKind::IP;
	}
	throw BinderException("vindex: unknown metric '%s' (expected 'l2sq', 'cosine', or 'ip')", raw);
}

idx_t ParseRerankMultiple(const case_insensitive_map_t<Value> &options) {
	auto it = options.find("rerank");
	if (it == options.end()) {
		return 1;
	}
	const int64_t v = it->second.GetValue<int64_t>();
	if (v < 1) {
		throw BinderException("vindex: 'rerank' must be >= 1 (got %lld)", (long long)v);
	}
	if (v > int64_t(UINT32_MAX)) {
		throw BinderException("vindex: 'rerank' is unreasonably large (%lld)", (long long)v);
	}
	return idx_t(v);
}

HnswCoreParams ParseCoreParams(const case_insensitive_map_t<Value> &options, idx_t dim) {
	HnswCoreParams p;
	p.dim = dim;
	auto m_opt = options.find("m");
	if (m_opt != options.end()) {
		const int32_t v = m_opt->second.GetValue<int32_t>();
		if (v <= 0 || v > UINT16_MAX) {
			throw BinderException("vindex: 'm' must be in (0, %d]", int(UINT16_MAX));
		}
		p.m = uint16_t(v);
		// Classical: m0 = 2*m unless caller overrides.
		p.m0 = uint16_t(std::min<int32_t>(v * 2, UINT16_MAX));
	}
	auto m0_opt = options.find("m0");
	if (m0_opt != options.end()) {
		const int32_t v = m0_opt->second.GetValue<int32_t>();
		if (v <= 0 || v > UINT16_MAX) {
			throw BinderException("vindex: 'm0' must be in (0, %d]", int(UINT16_MAX));
		}
		p.m0 = uint16_t(v);
	}
	auto efc_opt = options.find("ef_construction");
	if (efc_opt != options.end()) {
		const int32_t v = efc_opt->second.GetValue<int32_t>();
		if (v <= 0 || v > UINT16_MAX) {
			throw BinderException("vindex: 'ef_construction' must be in (0, %d]", int(UINT16_MAX));
		}
		p.ef_construction = uint16_t(v);
	}
	auto efs_opt = options.find("ef_search");
	if (efs_opt != options.end()) {
		const int32_t v = efs_opt->second.GetValue<int32_t>();
		if (v <= 0 || v > UINT16_MAX) {
			throw BinderException("vindex: 'ef_search' must be in (0, %d]", int(UINT16_MAX));
		}
		p.ef_search = uint16_t(v);
	}
	return p;
}

// Layout of the state stream written by WriteStateStream / read by
// ReadStateStream:
//
//   u64 magic                      ("VNDXHNW2")
//   u64 rerank_multiple
//   u32 quantizer_blob_size
//   u8[]  quantizer_blob           (quantizer->Serialize())
//   u32 core_blob_size
//   u8[]  core_blob                (HnswCore::SerializeState())
//   u64 row_to_block_size
//   { i64 row_id, u64 block_id } × row_to_block_size
//   u64 tombstones_size
//   { i64 row_id } × tombstones_size
//
// "VNDXHNW1" (pre-M1.6e) indices have no rerank field; we fall back to
// rerank_multiple=1 when we see it.
constexpr uint64_t kStateMagicV1 = 0x564E4458484E5731ULL; // "VNDXHNW1"
constexpr uint64_t kStateMagicV2 = 0x564E4458484E5732ULL; // "VNDXHNW2"

} // namespace

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

void HnswIndex::InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size, MetricKind metric) {
	metric_ = metric;
	dim_ = vector_size;
	params_ = ParseCoreParams(options, vector_size);
	rerank_multiple_ = ParseRerankMultiple(options);
	quantizer_ = CreateQuantizer(options, metric_, dim_);
	core_ = make_uniq<HnswCore>(params_, *quantizer_, *store_);
}

HnswIndex::HnswIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
                     TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                     AttachedDatabase &db, const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                     idx_t estimated_cardinality)
    : VectorIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("HNSW indexes do not support unique or primary key constraints");
	}

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	store_ = make_uniq<IndexBlockStore>(block_manager);

	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);
	const auto vector_size = ArrayType::GetSize(vector_type);
	const auto vector_child_type = ArrayType::GetChildType(vector_type);
	if (vector_child_type.id() != LogicalTypeId::FLOAT) {
		throw NotImplementedException("HNSW indexes currently support only FLOAT[] vectors");
	}

	const auto metric = ParseMetric(options);
	InitFromOptions(options, vector_size, metric);

	auto lock = rwlock.GetExclusiveLock();
	if (info.IsValid()) {
		state_root_.Set(info.root);
		// Re-register node allocators to match what the serialized index used.
		// HnswCore's constructor has already done so for the max_level + 1 sizes
		// it needs, so the only thing to restore is the on-disk allocator state.
		store_->Init(info);
		if (state_root_.Get() != 0) {
			ReadStateStream(state_root_);
		}
	}
	index_size = core_->Size();
	(void)estimated_cardinality;
}

//------------------------------------------------------------------------------
// State stream helpers
//------------------------------------------------------------------------------

void HnswIndex::WriteStateStream() {
	if (state_root_.Get() == 0) {
		state_root_ = BlockId {};
	}
	auto writer = store_->BeginStream(state_root_);
	state_root_ = writer->Root();

	const uint64_t magic = kStateMagicV2;
	writer->Write(reinterpret_cast<const_data_ptr_t>(&magic), sizeof(magic));

	const uint64_t rerank = uint64_t(rerank_multiple_);
	writer->Write(reinterpret_cast<const_data_ptr_t>(&rerank), sizeof(rerank));

	vector<data_t> qblob;
	quantizer_->Serialize(qblob);
	const uint32_t qsize = uint32_t(qblob.size());
	writer->Write(reinterpret_cast<const_data_ptr_t>(&qsize), sizeof(qsize));
	if (qsize > 0) {
		writer->Write(qblob.data(), qsize);
	}

	vector<data_t> cblob;
	core_->SerializeState(cblob);
	const uint32_t csize = uint32_t(cblob.size());
	writer->Write(reinterpret_cast<const_data_ptr_t>(&csize), sizeof(csize));
	if (csize > 0) {
		writer->Write(cblob.data(), csize);
	}

	const uint64_t rsize = row_to_block_.size();
	writer->Write(reinterpret_cast<const_data_ptr_t>(&rsize), sizeof(rsize));
	for (const auto &kv : row_to_block_) {
		const int64_t rid = kv.first;
		const uint64_t bid = kv.second.Get();
		writer->Write(reinterpret_cast<const_data_ptr_t>(&rid), sizeof(rid));
		writer->Write(reinterpret_cast<const_data_ptr_t>(&bid), sizeof(bid));
	}

	const uint64_t tsize = tombstones_.size();
	writer->Write(reinterpret_cast<const_data_ptr_t>(&tsize), sizeof(tsize));
	for (const auto &t : tombstones_) {
		const int64_t rid = t;
		writer->Write(reinterpret_cast<const_data_ptr_t>(&rid), sizeof(rid));
	}
}

void HnswIndex::ReadStateStream(BlockId root) {
	auto reader = store_->OpenStream(root);

	uint64_t magic = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&magic), sizeof(magic));
	if (magic != kStateMagicV1 && magic != kStateMagicV2) {
		throw InternalException("HnswIndex: unrecognized state stream (magic mismatch)");
	}

	if (magic == kStateMagicV2) {
		uint64_t rerank = 1;
		reader->Read(reinterpret_cast<data_ptr_t>(&rerank), sizeof(rerank));
		rerank_multiple_ = idx_t(rerank == 0 ? 1 : rerank);
	} else {
		rerank_multiple_ = 1;
	}

	uint32_t qsize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&qsize), sizeof(qsize));
	if (qsize > 0) {
		vector<data_t> qblob(qsize);
		reader->Read(qblob.data(), qsize);
		quantizer_->Deserialize(qblob.data(), qsize);
	}

	uint32_t csize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&csize), sizeof(csize));
	if (csize > 0) {
		vector<data_t> cblob(csize);
		reader->Read(cblob.data(), csize);
		core_->DeserializeState(cblob.data(), csize);
	}

	uint64_t rsize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&rsize), sizeof(rsize));
	row_to_block_.clear();
	row_to_block_.reserve(rsize);
	for (uint64_t i = 0; i < rsize; i++) {
		int64_t rid = 0;
		uint64_t bid = 0;
		reader->Read(reinterpret_cast<data_ptr_t>(&rid), sizeof(rid));
		reader->Read(reinterpret_cast<data_ptr_t>(&bid), sizeof(bid));
		BlockId b;
		b.Set(bid);
		row_to_block_.emplace(row_t(rid), b);
	}

	uint64_t tsize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&tsize), sizeof(tsize));
	tombstones_.clear();
	tombstones_.reserve(tsize);
	for (uint64_t i = 0; i < tsize; i++) {
		int64_t rid = 0;
		reader->Read(reinterpret_cast<data_ptr_t>(&rid), sizeof(rid));
		tombstones_.insert(row_t(rid));
	}
}

//------------------------------------------------------------------------------
// Metadata
//------------------------------------------------------------------------------

idx_t HnswIndex::GetVectorSize() const {
	return dim_;
}

MetricKind HnswIndex::GetMetricKind() const {
	return metric_;
}

idx_t HnswIndex::GetRerankMultiple(ClientContext &context) const {
	Value v;
	if (context.TryGetCurrentSetting("vindex_rerank_multiple", v)) {
		if (!v.IsNull() && v.type() == LogicalType::BIGINT) {
			const auto session_val = v.GetValue<int64_t>();
			if (session_val > 0) {
				return idx_t(session_val);
			}
		}
	}
	return rerank_multiple_ == 0 ? 1 : rerank_multiple_;
}

unique_ptr<HnswIndexStats> HnswIndex::GetStats() {
	auto lock = rwlock.GetExclusiveLock();
	auto result = make_uniq<HnswIndexStats>();
	result->count = core_->Size();
	result->max_level = core_->MaxLevel();
	// Without usearch's internal capacity concept we just report count; the
	// allocator grows on demand via FixedSizeAllocator anyway.
	result->capacity = result->count;
	result->approx_size = store_->GetInMemorySize();

	// Per-level stats: traversing every node is O(N) — acceptable since this
	// pragma is intended for inspection, not hot-path telemetry.
	vector<HnswLevelStats> per_level(idx_t(params_.max_level) + 1);
	for (auto &kv : row_to_block_) {
		if (tombstones_.count(kv.first)) {
			continue;
		}
		// We can't cheaply ask HnswCore for per-node level without Pinning,
		// and the node layout is private. Skip the detailed breakdown for M1:
		// report only layer 0 (every node participates there) and layer info
		// from MaxLevel. This keeps the schema byte-compatible with the old
		// usearch path while avoiding leaking HnswCore internals.
		per_level[0].nodes++;
	}
	for (idx_t L = 0; L <= params_.max_level; L++) {
		per_level[L].max_edges = L == 0 ? params_.m0 : params_.m;
	}
	for (auto &s : per_level) {
		result->level_stats.push_back(s);
	}
	return result;
}

//------------------------------------------------------------------------------
// Scans
//------------------------------------------------------------------------------

struct HnswIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	unique_array<row_t> row_ids = nullptr;
};

static idx_t ResolveEfSearch(ClientContext &context, idx_t fallback) {
	Value ef_opt;
	if (context.TryGetCurrentSetting("vindex_ef_search", ef_opt) ||
	    context.TryGetCurrentSetting("hnsw_ef_search", ef_opt)) {
		if (!ef_opt.IsNull() && ef_opt.type() == LogicalType::BIGINT) {
			const auto val = ef_opt.GetValue<int64_t>();
			if (val > 0) {
				return static_cast<idx_t>(val);
			}
		}
	}
	return fallback;
}

unique_ptr<IndexScanState> HnswIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<HnswIndexScanState>();
	const idx_t ef_search = ResolveEfSearch(context, params_.ef_search);

	vector<float> query_workspace(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, query_workspace.data());

	auto lock = rwlock.GetSharedLock();
	// Over-fetch to compensate for tombstones: worst case we need `limit` live
	// hits, so grabbing `limit + tombstones_.size()` from the graph is tight.
	const idx_t oversample = limit + tombstones_.size();
	auto cands = core_->Search(query_workspace.data(), oversample, ef_search);

	vector<row_t> hits;
	hits.reserve(limit);
	for (auto &c : cands) {
		if (hits.size() >= limit) {
			break;
		}
		if (tombstones_.count(row_t(c.row_id))) {
			continue;
		}
		hits.push_back(row_t(c.row_id));
	}

	state->current_row = 0;
	state->total_rows = hits.size();
	state->row_ids = make_uniq_array<row_t>(hits.size());
	for (idx_t i = 0; i < hits.size(); i++) {
		state->row_ids[i] = hits[i];
	}
	return std::move(state);
}

idx_t HnswIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<HnswIndexScanState>();

	idx_t count = 0;
	auto row_ids = FlatVector::GetDataMutable<row_t>(result) + result_offset;
	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}
	return count;
}

struct MultiScanState final : IndexScanState {
	Vector vec;
	vector<row_t> row_ids;
	idx_t ef_search;
	explicit MultiScanState(idx_t ef_search_p) : vec(LogicalType::ROW_TYPE, nullptr), ef_search(ef_search_p) {
	}
};

unique_ptr<IndexScanState> HnswIndex::InitializeMultiScan(ClientContext &context) {
	const idx_t ef_search = ResolveEfSearch(context, params_.ef_search);
	return make_uniq<MultiScanState>(ef_search);
}

idx_t HnswIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<MultiScanState>();

	vector<float> query_workspace(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, query_workspace.data());

	vector<HnswCore::Candidate> cands;
	{
		auto lock = rwlock.GetSharedLock();
		const idx_t oversample = limit + tombstones_.size();
		cands = core_->Search(query_workspace.data(), oversample, state.ef_search);
	}

	const auto offset = state.row_ids.size();
	for (auto &c : cands) {
		if (state.row_ids.size() - offset >= limit) {
			break;
		}
		if (tombstones_.count(row_t(c.row_id))) {
			continue;
		}
		state.row_ids.push_back(row_t(c.row_id));
	}
	return state.row_ids.size() - offset;
}

const Vector &HnswIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<MultiScanState>();
	FlatVector::SetData(scan_state.vec, (data_ptr_t)scan_state.row_ids.data(), count_t(scan_state.row_ids.size()));
	return scan_state.vec;
}

void HnswIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<MultiScanState>();
	scan_state.row_ids.clear();
}

//------------------------------------------------------------------------------
// Construction + mutation
//------------------------------------------------------------------------------

void HnswIndex::ResetStorage(IndexLock &index_lock) {
	auto lock = rwlock.GetExclusiveLock();
	// Drop everything the index owns. Metadata accessors (metric_, dim_,
	// params_) stay valid — usearch's old behavior kept them readable after
	// drop and DuckDB catalog callbacks rely on that.
	core_.reset();
	row_to_block_.clear();
	tombstones_.clear();
	state_root_.Clear();
	store_->Reset();
	// Rebuild an empty core so subsequent accessors don't hit a null dereference.
	// quantizer_ is reusable — Serialize/Deserialize is idempotent at this point.
	core_ = make_uniq<HnswCore>(params_, *quantizer_, *store_);
	index_size = 0;
}

void HnswIndex::TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap) {
	// Shortcut for quantizers that don't need training (FlatQuantizer) —
	// the virtual call is cheap and this avoids materializing the sample
	// buffer for no reason. We always pass through at least once to keep the
	// contract "Train is called before Encode" explicit.
	auto lock = rwlock.GetExclusiveLock();

	const idx_t total = collection.Count();
	const idx_t n = std::min<idx_t>(total, sample_cap);
	vector<float> buf;
	if (n > 0) {
		buf.resize(n * dim_);

		DataChunk scan_chunk;
		collection.InitializeScanChunk(scan_chunk);
		ColumnDataScanState scan_state;
		// Only read the vector column (index 0); row_id column is not needed
		// for training.
		collection.InitializeScan(scan_state, {0}, ColumnDataScanProperties::ALLOW_ZERO_COPY);

		idx_t copied = 0;
		while (copied < n && collection.Scan(scan_state, scan_chunk)) {
			const auto chunk_size = scan_chunk.size();
			if (chunk_size == 0) {
				continue;
			}
			scan_chunk.Flatten();
			auto &vec_vec = scan_chunk.data[0];
			auto &child_vec = ArrayVector::GetEntry(vec_vec);
			auto *vec_data = FlatVector::GetData<float>(child_vec);
			const idx_t take = std::min<idx_t>(chunk_size, n - copied);
			for (idx_t i = 0; i < take; i++) {
				if (FlatVector::IsNull(vec_vec, i)) {
					// Treat null rows as zero vectors; they won't be inserted
					// into the graph anyway, this just keeps the sample padded.
					std::memset(buf.data() + (copied + i) * dim_, 0, dim_ * sizeof(float));
				} else {
					std::memcpy(buf.data() + (copied + i) * dim_, vec_data + i * dim_, dim_ * sizeof(float));
				}
			}
			copied += take;
		}
	}

	quantizer_->Train(buf.data(), n, dim_);
}

void HnswIndex::Construct(DataChunk &input, Vector &row_ids, idx_t /*thread_idx*/) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());
	is_dirty = true;

	const auto count = input.size();
	input.Flatten();

	auto &vec_vec = input.data[0];
	auto &vec_child_vec = ArrayVector::GetEntry(vec_vec);
	const auto array_size = ArrayType::GetSize(vec_vec.GetType());
	auto vec_child_data = FlatVector::GetData<float>(vec_child_vec);
	auto rowid_data = FlatVector::GetData<row_t>(row_ids);

	// HnswCore is not internally thread-safe. All writers serialize through
	// the exclusive lock — the thread_idx argument is ignored. See README
	// §"Why not usearch" for the tradeoff.
	auto lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(vec_vec, i)) {
			continue;
		}
		const auto rowid = rowid_data[i];
		// Resurrect a tombstoned row_id on re-insert (mirrors usearch's
		// remove/add-same-key behavior). The underlying node stays; we just
		// drop the tombstone and update row_to_block.
		auto tomb_it = tombstones_.find(rowid);
		if (tomb_it != tombstones_.end()) {
			tombstones_.erase(tomb_it);
		}
		const BlockId b = core_->Insert(int64_t(rowid), vec_child_data + (i * array_size));
		row_to_block_[rowid] = b;
	}
	index_size = core_->Size();
}

void HnswIndex::Compact() {
	is_dirty = true;
	// Tombstone compaction would require either (a) a HnswCore::Remove that
	// rewrites every edge pointing at the removed node — expensive and subtle,
	// or (b) a full rebuild from row_id → original vector, which we cannot
	// reconstruct once RaBitQ encoding lossy-compresses the codes (M1.6e).
	//
	// For M1 this is a no-op: the Scan path already filters tombstones out,
	// and in practice delete-heavy workloads should rebuild the index. We keep
	// the PRAGMA signature so hnsw_crud.test still passes.
	auto lock = rwlock.GetExclusiveLock();
}

void HnswIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	is_dirty = true;
	const auto count = input.size();
	rowid_vec.Flatten(count);
	auto row_id_data = FlatVector::GetData<row_t>(rowid_vec);

	auto _lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		const auto rid = row_id_data[i];
		if (row_to_block_.count(rid)) {
			tombstones_.insert(rid);
		}
	}
}

ErrorData HnswIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	Construct(input, rowid_vec, 0);
	return ErrorData {};
}

ErrorData HnswIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(appended_data, expression_result);
	Construct(expression_result, row_identifiers, 0);
	return ErrorData {};
}

//------------------------------------------------------------------------------
// Persistence
//------------------------------------------------------------------------------

void HnswIndex::PersistToDisk() {
	auto lock = rwlock.GetExclusiveLock();
	if (!is_dirty) {
		return;
	}
	WriteStateStream();
	is_dirty = false;
}

IndexStorageInfo HnswIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	store_->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();

	IndexStorageInfo info = store_->GetInfo();
	info.name = name;
	info.root = state_root_.Get();
	return info;
}

IndexStorageInfo HnswIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info = store_->GetInfo();
	info.name = name;
	info.root = state_root_.Get();
	info.buffers = store_->InitSerializationToWAL();
	return info;
}

idx_t HnswIndex::GetInMemorySize(IndexLock &state) {
	return store_->GetInMemorySize();
}

bool HnswIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("HnswIndex::MergeIndexes() not implemented");
}

void HnswIndex::Vacuum(IndexLock &state) {
}

void HnswIndex::Verify(IndexLock &state) {
}

string HnswIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("HnswIndex::ToString() not implemented");
}

void HnswIndex::VerifyAllocations(IndexLock &state) {
}

void HnswIndex::VerifyBuffers(IndexLock &lock) {
}

//------------------------------------------------------------------------------
// Register IndexType + settings
//------------------------------------------------------------------------------

void RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;
	index_type.name = HnswIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<HnswIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = HnswIndex::CreatePlan;

	if (!db.config.GetOptionByName("vindex_enable_experimental_persistence")) {
		db.config.AddExtensionOption("vindex_enable_experimental_persistence",
		                             "experimental: enable creating vector indexes in persistent databases",
		                             LogicalType::BOOLEAN, Value::BOOLEAN(false));
	}
	if (!db.config.GetOptionByName("hnsw_enable_experimental_persistence")) {
		db.config.AddExtensionOption("hnsw_enable_experimental_persistence",
		                             "deprecated alias of vindex_enable_experimental_persistence",
		                             LogicalType::BOOLEAN, Value::BOOLEAN(false));
	}

	if (!db.config.GetOptionByName("vindex_ef_search")) {
		db.config.AddExtensionOption("vindex_ef_search",
		                             "override the ef_search parameter when scanning HNSW indexes",
		                             LogicalType::BIGINT);
	}
	if (!db.config.GetOptionByName("vindex_rerank_multiple")) {
		db.config.AddExtensionOption("vindex_rerank_multiple",
		                             "override the rerank over-fetch factor when scanning vector indexes "
		                             "(index returns k × rerank_multiple candidates; upstream TopN/min_by "
		                             "reranks by exact distance)",
		                             LogicalType::BIGINT);
	}
	if (!db.config.GetOptionByName("hnsw_ef_search")) {
		db.config.AddExtensionOption("hnsw_ef_search", "deprecated alias of vindex_ef_search", LogicalType::BIGINT);
	}

	db.config.GetIndexTypes().RegisterIndexType(index_type);

	VectorIndexRegistry::Instance().RegisterTypeName(HnswIndex::TYPE_NAME);
}

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
