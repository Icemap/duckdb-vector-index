#include "algo/diskann/diskann_index.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/index/index_type_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "algo/diskann/diskann_module.hpp"
#include "vindex/vector_index_registry.hpp"

#include <cstring>

namespace duckdb {
namespace vindex {
namespace diskann {

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
	return idx_t(v);
}

DiskAnnCoreParams ParseCoreParams(const case_insensitive_map_t<Value> &options, idx_t dim) {
	DiskAnnCoreParams p;
	p.dim = dim;
	auto r_it = options.find("diskann_r");
	if (r_it != options.end()) {
		const int64_t v = r_it->second.GetValue<int64_t>();
		if (v < 4 || v > 256) {
			throw BinderException("DiskANN 'diskann_r' out of range (4..256)");
		}
		p.R = uint16_t(v);
	}
	auto l_it = options.find("diskann_l");
	if (l_it != options.end()) {
		const int64_t v = l_it->second.GetValue<int64_t>();
		if (v < 4 || v > 1024) {
			throw BinderException("DiskANN 'diskann_l' out of range (4..1024)");
		}
		p.L = uint16_t(v);
	}
	if (p.L < p.R) {
		p.L = p.R;
	}
	auto a_it = options.find("diskann_alpha");
	if (a_it != options.end()) {
		const double v = a_it->second.GetValue<double>();
		if (v < 1.0 || v > 2.0) {
			throw BinderException("DiskANN 'diskann_alpha' out of range (1.0..2.0)");
		}
		p.alpha = float(v);
	}
	return p;
}

constexpr uint64_t kStateMagic = 0x325644444E534956ULL; // "VISNDDDV2" — distinct from ivf/hnsw

idx_t SampleFloats(ColumnDataCollection &collection, idx_t dim, idx_t sample_cap, vector<float> &buf) {
	const idx_t total = collection.Count();
	const idx_t n = std::min<idx_t>(total, sample_cap);
	buf.clear();
	if (n == 0) {
		return 0;
	}
	buf.resize(n * dim);

	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);
	ColumnDataScanState scan_state;
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
				std::memset(buf.data() + (copied + i) * dim, 0, dim * sizeof(float));
			} else {
				std::memcpy(buf.data() + (copied + i) * dim, vec_data + i * dim, dim * sizeof(float));
			}
		}
		copied += take;
	}
	return copied;
}

} // namespace

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

void DiskAnnIndex::InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size,
                                   MetricKind metric) {
	metric_ = metric;
	dim_ = vector_size;
	params_ = ParseCoreParams(options, vector_size);
	rerank_multiple_ = ParseRerankMultiple(options);
	quantizer_ = CreateQuantizer(options, metric_, dim_);
	// Block inline-float quantizer: DiskANN's whole point is graph blocks that
	// can be evicted while codes stay in RAM. A FLAT quantizer puts the full
	// fp32 vector into codes_, which makes the index no cheaper than HNSW-Flat
	// and loses the >RAM benefit. Force the user to pick a compressing
	// quantizer so the storage story is coherent.
	if (quantizer_->Kind() == QuantizerKind::FLAT) {
		throw BinderException(
		    "DiskANN requires a compressing quantizer: use WITH (quantizer='pq') or (quantizer='rabitq'). "
		    "FLAT would store full fp32 codes in RAM and defeat DiskANN's >RAM layout.");
	}
	core_ = make_uniq<DiskAnnCore>(params_, *quantizer_, *store_);
}

DiskAnnIndex::DiskAnnIndex(const string &name, IndexConstraintType index_constraint_type,
                           const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                           const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                           const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                           idx_t estimated_cardinality)
    : VectorIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("DiskANN indexes do not support unique or primary key constraints");
	}

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	store_ = make_uniq<IndexBlockStore>(block_manager);

	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);
	const auto vector_size = ArrayType::GetSize(vector_type);
	const auto vector_child_type = ArrayType::GetChildType(vector_type);
	if (vector_child_type.id() != LogicalTypeId::FLOAT) {
		throw NotImplementedException("DiskANN indexes currently support only FLOAT[] vectors");
	}

	const auto metric = ParseMetric(options);
	InitFromOptions(options, vector_size, metric);

	auto lock = rwlock.GetExclusiveLock();
	if (info.IsValid()) {
		state_root_.Set(info.root);
		store_->Init(info);
		if (state_root_.Get() != 0) {
			ReadStateStream(state_root_);
		}
	}
	index_size = core_->Size();
	(void)estimated_cardinality;
}

//------------------------------------------------------------------------------
// State stream
//------------------------------------------------------------------------------

void DiskAnnIndex::WriteStateStream() {
	if (state_root_.Get() == 0) {
		state_root_ = BlockId {};
	}
	auto writer = store_->BeginStream(state_root_);
	state_root_ = writer->Root();

	const uint64_t magic = kStateMagic;
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

void DiskAnnIndex::ReadStateStream(BlockId root) {
	auto reader = store_->OpenStream(root);

	uint64_t magic = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&magic), sizeof(magic));
	if (magic != kStateMagic) {
		throw InternalException("DiskAnnIndex: unrecognized state stream (magic mismatch)");
	}

	uint64_t rerank = 1;
	reader->Read(reinterpret_cast<data_ptr_t>(&rerank), sizeof(rerank));
	rerank_multiple_ = idx_t(rerank == 0 ? 1 : rerank);

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
		uint64_t bid_raw = 0;
		reader->Read(reinterpret_cast<data_ptr_t>(&rid), sizeof(rid));
		reader->Read(reinterpret_cast<data_ptr_t>(&bid_raw), sizeof(bid_raw));
		BlockId bid;
		bid.Set(bid_raw);
		row_to_block_.emplace(row_t(rid), bid);
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

idx_t DiskAnnIndex::GetVectorSize() const {
	return dim_;
}

MetricKind DiskAnnIndex::GetMetricKind() const {
	return metric_;
}

idx_t DiskAnnIndex::GetRerankMultiple(ClientContext &context) const {
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

//------------------------------------------------------------------------------
// Scans
//------------------------------------------------------------------------------

struct DiskAnnIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	unique_array<row_t> row_ids = nullptr;
};

static idx_t ResolveLSearch(ClientContext &context, idx_t fallback) {
	Value v;
	if (context.TryGetCurrentSetting("vindex_diskann_l_search", v)) {
		if (!v.IsNull() && v.type() == LogicalType::BIGINT) {
			const auto val = v.GetValue<int64_t>();
			if (val > 0) {
				return idx_t(val);
			}
		}
	}
	return fallback;
}

unique_ptr<IndexScanState> DiskAnnIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<DiskAnnIndexScanState>();
	const idx_t L_search = ResolveLSearch(context, params_.L);

	vector<float> query_workspace(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, query_workspace.data());

	auto lock = rwlock.GetSharedLock();
	const idx_t oversample = limit + tombstones_.size();
	auto cands = core_->Search(query_workspace.data(), oversample, L_search);

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

idx_t DiskAnnIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<DiskAnnIndexScanState>();
	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;
	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}
	return count;
}

struct DiskAnnMultiScanState final : IndexScanState {
	Vector vec;
	vector<row_t> row_ids;
	idx_t L_search;
	explicit DiskAnnMultiScanState(idx_t L_p) : vec(LogicalType::ROW_TYPE, nullptr), L_search(L_p) {
	}
};

unique_ptr<IndexScanState> DiskAnnIndex::InitializeMultiScan(ClientContext &context) {
	const idx_t L_search = ResolveLSearch(context, params_.L);
	return make_uniq<DiskAnnMultiScanState>(L_search);
}

idx_t DiskAnnIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<DiskAnnMultiScanState>();
	vector<float> query_workspace(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, query_workspace.data());

	vector<DiskAnnCore::Candidate> cands;
	{
		auto lock = rwlock.GetSharedLock();
		const idx_t oversample = limit + tombstones_.size();
		cands = core_->Search(query_workspace.data(), oversample, state.L_search);
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

const Vector &DiskAnnIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<DiskAnnMultiScanState>();
	FlatVector::SetData(scan_state.vec, (data_ptr_t)scan_state.row_ids.data());
	return scan_state.vec;
}

void DiskAnnIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<DiskAnnMultiScanState>();
	scan_state.row_ids.clear();
}

//------------------------------------------------------------------------------
// Construction + mutation
//------------------------------------------------------------------------------

void DiskAnnIndex::CommitDrop(IndexLock &index_lock) {
	auto lock = rwlock.GetExclusiveLock();
	core_.reset();
	row_to_block_.clear();
	tombstones_.clear();
	state_root_.Clear();
	store_->Reset();
	core_ = make_uniq<DiskAnnCore>(params_, *quantizer_, *store_);
	index_size = 0;
}

void DiskAnnIndex::TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap) {
	auto lock = rwlock.GetExclusiveLock();
	vector<float> buf;
	const idx_t n = SampleFloats(collection, dim_, sample_cap, buf);
	quantizer_->Train(buf.data(), n, dim_);
}

void DiskAnnIndex::Construct(DataChunk &input, Vector &row_ids, idx_t /*thread_idx*/) {
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

	// DiskAnnCore::Insert is not internally thread-safe (beam search + edge
	// updates share one rng + visit table). Serialize at the index level.
	auto lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(vec_vec, i)) {
			continue;
		}
		const auto rowid = rowid_data[i];
		auto tomb_it = tombstones_.find(rowid);
		if (tomb_it != tombstones_.end()) {
			tombstones_.erase(tomb_it);
		}
		const BlockId bid = core_->Insert(int64_t(rowid), vec_child_data + (i * array_size));
		row_to_block_[rowid] = bid;
	}
	index_size = core_->Size();
}

void DiskAnnIndex::Compact() {
	is_dirty = true;
	auto lock = rwlock.GetExclusiveLock();
	// Like IVF/HNSW: tombstones filter results at query time; full compaction
	// needs a rebuild, which must go through CREATE INDEX.
}

void DiskAnnIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
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

ErrorData DiskAnnIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	Construct(input, rowid_vec, 0);
	return ErrorData {};
}

ErrorData DiskAnnIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(appended_data, expression_result);
	Construct(expression_result, row_identifiers, 0);
	return ErrorData {};
}

//------------------------------------------------------------------------------
// Persistence
//------------------------------------------------------------------------------

void DiskAnnIndex::PersistToDisk() {
	auto lock = rwlock.GetExclusiveLock();
	if (!is_dirty) {
		return;
	}
	WriteStateStream();
	is_dirty = false;
}

IndexStorageInfo DiskAnnIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
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

IndexStorageInfo DiskAnnIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	PersistToDisk();
	IndexStorageInfo info = store_->GetInfo();
	info.name = name;
	info.root = state_root_.Get();
	info.buffers = store_->InitSerializationToWAL();
	return info;
}

idx_t DiskAnnIndex::GetInMemorySize(IndexLock &state) {
	return store_->GetInMemorySize();
}

bool DiskAnnIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("DiskAnnIndex::MergeIndexes() not implemented");
}

void DiskAnnIndex::Vacuum(IndexLock &state) {
}

void DiskAnnIndex::Verify(IndexLock &state) {
}

string DiskAnnIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("DiskAnnIndex::ToString() not implemented");
}

void DiskAnnIndex::VerifyAllocations(IndexLock &state) {
}

void DiskAnnIndex::VerifyBuffers(IndexLock &lock) {
}

//------------------------------------------------------------------------------
// Registration
//------------------------------------------------------------------------------

void RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;
	index_type.name = DiskAnnIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<DiskAnnIndex>(input.name, input.constraint_type, input.column_ids,
		                                   input.table_io_manager, input.unbound_expressions, input.db,
		                                   input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = DiskAnnIndex::CreatePlan;

	if (!db.config.GetOptionByName("vindex_enable_experimental_persistence")) {
		db.config.AddExtensionOption("vindex_enable_experimental_persistence",
		                             "experimental: enable creating vector indexes in persistent databases",
		                             LogicalType::BOOLEAN, Value::BOOLEAN(false));
	}
	if (!db.config.GetOptionByName("vindex_rerank_multiple")) {
		db.config.AddExtensionOption("vindex_rerank_multiple",
		                             "override the rerank over-fetch factor when scanning vector indexes",
		                             LogicalType::BIGINT);
	}
	if (!db.config.GetOptionByName("vindex_diskann_l_search")) {
		db.config.AddExtensionOption("vindex_diskann_l_search",
		                             "override the DiskANN beam width (L_search) when scanning",
		                             LogicalType::BIGINT);
	}

	db.config.GetIndexTypes().RegisterIndexType(index_type);

	VectorIndexRegistry::Instance().RegisterTypeName(DiskAnnIndex::TYPE_NAME);
}

} // namespace diskann
} // namespace vindex
} // namespace duckdb
