#include "algo/ivf/ivf_index.hpp"

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

#include "algo/ivf/ivf_module.hpp"
#include "vindex/vector_index_registry.hpp"

#include <cstring>

namespace duckdb {
namespace vindex {
namespace ivf {

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

IvfCoreParams ParseCoreParams(const case_insensitive_map_t<Value> &options, idx_t dim) {
	IvfCoreParams p;
	p.dim = dim;
	auto nlist_it = options.find("nlist");
	if (nlist_it != options.end()) {
		const int64_t v = nlist_it->second.GetValue<int64_t>();
		if (v < 2) {
			throw BinderException("IVF index 'nlist' must be >= 2 (got %lld)", (long long)v);
		}
		p.nlist = idx_t(v);
	}
	auto nprobe_it = options.find("nprobe");
	if (nprobe_it != options.end()) {
		const int64_t v = nprobe_it->second.GetValue<int64_t>();
		if (v < 1) {
			throw BinderException("IVF index 'nprobe' must be >= 1 (got %lld)", (long long)v);
		}
		p.nprobe = idx_t(v);
	}
	if (p.nprobe > p.nlist) {
		p.nprobe = p.nlist;
	}
	return p;
}

// State stream layout:
//   u64 magic                     ("VNDXIVF1")
//   u64 rerank_multiple
//   u32 quantizer_blob_size
//   u8[] quantizer_blob
//   u32 core_blob_size
//   u8[] core_blob
//   u64 row_to_centroid_size
//   { i64 row_id, u64 centroid } × row_to_centroid_size
//   u64 tombstones_size
//   { i64 row_id } × tombstones_size
constexpr uint64_t kStateMagic = 0x315656494E444E56ULL; // "VNDXIVF1" — matches ivf_core.cpp

// Sample helper: read up to `sample_cap` float rows from column 0 of the
// collection into `buf`. Shared by TrainQuantizer / TrainCentroids so the
// two training passes see the same sample size cap but independent copies.
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

void IvfIndex::InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size, MetricKind metric) {
	metric_ = metric;
	dim_ = vector_size;
	params_ = ParseCoreParams(options, vector_size);
	rerank_multiple_ = ParseRerankMultiple(options);
	quantizer_ = CreateQuantizer(options, metric_, dim_);
	core_ = make_uniq<IvfCore>(params_, *quantizer_, *store_);
}

IvfIndex::IvfIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
                   TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                   AttachedDatabase &db, const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                   idx_t estimated_cardinality)
    : VectorIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("IVF indexes do not support unique or primary key constraints");
	}

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	store_ = make_uniq<IndexBlockStore>(block_manager);

	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);
	const auto vector_size = ArrayType::GetSize(vector_type);
	const auto vector_child_type = ArrayType::GetChildType(vector_type);
	if (vector_child_type.id() != LogicalTypeId::FLOAT) {
		throw NotImplementedException("IVF indexes currently support only FLOAT[] vectors");
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
// State stream helpers
//------------------------------------------------------------------------------

void IvfIndex::WriteStateStream() {
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

	const uint64_t rsize = row_to_centroid_.size();
	writer->Write(reinterpret_cast<const_data_ptr_t>(&rsize), sizeof(rsize));
	for (const auto &kv : row_to_centroid_) {
		const int64_t rid = kv.first;
		const uint64_t cid = kv.second;
		writer->Write(reinterpret_cast<const_data_ptr_t>(&rid), sizeof(rid));
		writer->Write(reinterpret_cast<const_data_ptr_t>(&cid), sizeof(cid));
	}

	const uint64_t tsize = tombstones_.size();
	writer->Write(reinterpret_cast<const_data_ptr_t>(&tsize), sizeof(tsize));
	for (const auto &t : tombstones_) {
		const int64_t rid = t;
		writer->Write(reinterpret_cast<const_data_ptr_t>(&rid), sizeof(rid));
	}
}

void IvfIndex::ReadStateStream(BlockId root) {
	auto reader = store_->OpenStream(root);

	uint64_t magic = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&magic), sizeof(magic));
	if (magic != kStateMagic) {
		throw InternalException("IvfIndex: unrecognized state stream (magic mismatch)");
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
	row_to_centroid_.clear();
	row_to_centroid_.reserve(rsize);
	for (uint64_t i = 0; i < rsize; i++) {
		int64_t rid = 0;
		uint64_t cid = 0;
		reader->Read(reinterpret_cast<data_ptr_t>(&rid), sizeof(rid));
		reader->Read(reinterpret_cast<data_ptr_t>(&cid), sizeof(cid));
		row_to_centroid_.emplace(row_t(rid), idx_t(cid));
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

idx_t IvfIndex::GetVectorSize() const {
	return dim_;
}

MetricKind IvfIndex::GetMetricKind() const {
	return metric_;
}

idx_t IvfIndex::GetRerankMultiple(ClientContext &context) const {
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

struct IvfIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	unique_array<row_t> row_ids = nullptr;
};

static idx_t ResolveNprobe(ClientContext &context, idx_t fallback) {
	Value v;
	if (context.TryGetCurrentSetting("vindex_nprobe", v)) {
		if (!v.IsNull() && v.type() == LogicalType::BIGINT) {
			const auto val = v.GetValue<int64_t>();
			if (val > 0) {
				return idx_t(val);
			}
		}
	}
	return fallback;
}

unique_ptr<IndexScanState> IvfIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<IvfIndexScanState>();
	const idx_t nprobe = ResolveNprobe(context, params_.nprobe);

	vector<float> query_workspace(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, query_workspace.data());

	auto lock = rwlock.GetSharedLock();
	const idx_t oversample = limit + tombstones_.size();
	auto cands = core_->Search(query_vector, query_workspace.data(), oversample, nprobe);

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

idx_t IvfIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<IvfIndexScanState>();
	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;
	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}
	return count;
}

struct IvfMultiScanState final : IndexScanState {
	Vector vec;
	vector<row_t> row_ids;
	idx_t nprobe;
	explicit IvfMultiScanState(idx_t nprobe_p) : vec(LogicalType::ROW_TYPE, nullptr), nprobe(nprobe_p) {
	}
};

unique_ptr<IndexScanState> IvfIndex::InitializeMultiScan(ClientContext &context) {
	const idx_t nprobe = ResolveNprobe(context, params_.nprobe);
	return make_uniq<IvfMultiScanState>(nprobe);
}

idx_t IvfIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<IvfMultiScanState>();

	vector<float> query_workspace(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, query_workspace.data());

	vector<IvfCore::Candidate> cands;
	{
		auto lock = rwlock.GetSharedLock();
		const idx_t oversample = limit + tombstones_.size();
		cands = core_->Search(query_vector, query_workspace.data(), oversample, state.nprobe);
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

const Vector &IvfIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<IvfMultiScanState>();
	FlatVector::SetData(scan_state.vec, (data_ptr_t)scan_state.row_ids.data());
	return scan_state.vec;
}

void IvfIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<IvfMultiScanState>();
	scan_state.row_ids.clear();
}

//------------------------------------------------------------------------------
// Construction + mutation
//------------------------------------------------------------------------------

void IvfIndex::CommitDrop(IndexLock &index_lock) {
	auto lock = rwlock.GetExclusiveLock();
	core_.reset();
	row_to_centroid_.clear();
	tombstones_.clear();
	state_root_.Clear();
	store_->Reset();
	core_ = make_uniq<IvfCore>(params_, *quantizer_, *store_);
	index_size = 0;
}

void IvfIndex::TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap) {
	auto lock = rwlock.GetExclusiveLock();
	vector<float> buf;
	const idx_t n = SampleFloats(collection, dim_, sample_cap, buf);
	quantizer_->Train(buf.data(), n, dim_);
}

void IvfIndex::TrainCentroids(ColumnDataCollection &collection, idx_t sample_cap) {
	auto lock = rwlock.GetExclusiveLock();
	vector<float> buf;
	const idx_t n = SampleFloats(collection, dim_, sample_cap, buf);
	if (n < params_.nlist) {
		// Too few samples to produce nlist unique centroids. KMeansPlusPlus
		// tolerates this and repeats samples, but it means the posting-list
		// distribution will be extremely uneven — warn-loudly is overkill
		// because the test fixtures frequently trigger it. Proceed silently.
	}
	core_->Train(buf.data(), n);
}

void IvfIndex::Construct(DataChunk &input, Vector &row_ids, idx_t /*thread_idx*/) {
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

	// IvfCore::Insert appends to one of nlist posting-list streams. The streams
	// share the IndexBlockStore's StreamWriter allocator, which is not
	// thread-safe in M1 — serialize at the index level.
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
		const idx_t c = core_->Insert(int64_t(rowid), vec_child_data + (i * array_size));
		row_to_centroid_[rowid] = c;
	}
	index_size = core_->Size();
}

void IvfIndex::Compact() {
	is_dirty = true;
	// Analogous to HnswIndex::Compact: rebuilding posting lists from
	// tombstones would require decoding every code or persisting originals,
	// neither of which we have. Scan filters tombstones; callers that need
	// tighter storage should rebuild.
	auto lock = rwlock.GetExclusiveLock();
}

void IvfIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	is_dirty = true;
	const auto count = input.size();
	rowid_vec.Flatten(count);
	auto row_id_data = FlatVector::GetData<row_t>(rowid_vec);

	auto _lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		const auto rid = row_id_data[i];
		if (row_to_centroid_.count(rid)) {
			tombstones_.insert(rid);
		}
	}
}

ErrorData IvfIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	Construct(input, rowid_vec, 0);
	return ErrorData {};
}

ErrorData IvfIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(appended_data, expression_result);
	Construct(expression_result, row_identifiers, 0);
	return ErrorData {};
}

//------------------------------------------------------------------------------
// Persistence
//------------------------------------------------------------------------------

void IvfIndex::PersistToDisk() {
	auto lock = rwlock.GetExclusiveLock();
	if (!is_dirty) {
		return;
	}
	WriteStateStream();
	is_dirty = false;
}

IndexStorageInfo IvfIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
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

IndexStorageInfo IvfIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	PersistToDisk();
	IndexStorageInfo info = store_->GetInfo();
	info.name = name;
	info.root = state_root_.Get();
	info.buffers = store_->InitSerializationToWAL();
	return info;
}

idx_t IvfIndex::GetInMemorySize(IndexLock &state) {
	return store_->GetInMemorySize();
}

bool IvfIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("IvfIndex::MergeIndexes() not implemented");
}

void IvfIndex::Vacuum(IndexLock &state) {
}

void IvfIndex::Verify(IndexLock &state) {
}

string IvfIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("IvfIndex::ToString() not implemented");
}

void IvfIndex::VerifyAllocations(IndexLock &state) {
}

void IvfIndex::VerifyBuffers(IndexLock &lock) {
}

//------------------------------------------------------------------------------
// Register IndexType + settings
//------------------------------------------------------------------------------

void RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;
	index_type.name = IvfIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		auto res = make_uniq<IvfIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                               input.unbound_expressions, input.db, input.options, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = IvfIndex::CreatePlan;

	// `vindex_enable_experimental_persistence` + `vindex_rerank_multiple` are
	// shared with HNSW — registered once by whichever algo runs first. Use
	// GetOptionByName to make the registration idempotent.
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
	if (!db.config.GetOptionByName("vindex_nprobe")) {
		db.config.AddExtensionOption("vindex_nprobe", "override the nprobe parameter when scanning IVF indexes",
		                             LogicalType::BIGINT);
	}

	db.config.GetIndexTypes().RegisterIndexType(index_type);

	VectorIndexRegistry::Instance().RegisterTypeName(IvfIndex::TYPE_NAME);
}

} // namespace ivf
} // namespace vindex
} // namespace duckdb
