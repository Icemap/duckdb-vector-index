#include "algo/hnsw/hnsw_index.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
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
#include "usearch/duckdb_usearch.hpp"

namespace duckdb {
namespace vindex {
namespace hnsw {

//------------------------------------------------------------------------------
// Linked Blocks (verbatim port from ref/duckdb-vss/src/hnsw/hnsw_index.cpp:45)
//------------------------------------------------------------------------------

class LinkedBlock {
public:
	static constexpr const idx_t BLOCK_SIZE = Storage::DEFAULT_BLOCK_SIZE - sizeof(validity_t);
	static constexpr const idx_t BLOCK_DATA_SIZE = BLOCK_SIZE - sizeof(IndexPointer);
	static_assert(BLOCK_SIZE > sizeof(IndexPointer), "Block size must be larger than the size of an IndexPointer");

	IndexPointer next_block;
	char data[BLOCK_DATA_SIZE] = {0};
};

constexpr idx_t LinkedBlock::BLOCK_DATA_SIZE;
constexpr idx_t LinkedBlock::BLOCK_SIZE;

class LinkedBlockReader {
private:
	FixedSizeAllocator &allocator;
	IndexPointer root_pointer;
	IndexPointer current_pointer;
	idx_t position_in_block;

public:
	LinkedBlockReader(FixedSizeAllocator &allocator, IndexPointer root_pointer)
	    : allocator(allocator), root_pointer(root_pointer), current_pointer(root_pointer), position_in_block(0) {
	}

	void Reset() {
		current_pointer = root_pointer;
		position_in_block = 0;
	}

	idx_t ReadData(data_ptr_t buffer, idx_t length) {
		idx_t bytes_read = 0;
		while (bytes_read < length) {
			auto block = allocator.Get<const LinkedBlock>(current_pointer, false);
			auto block_data = block->data;
			auto data_to_read = std::min(length - bytes_read, LinkedBlock::BLOCK_DATA_SIZE - position_in_block);
			std::memcpy(buffer + bytes_read, block_data + position_in_block, data_to_read);

			bytes_read += data_to_read;
			position_in_block += data_to_read;

			if (position_in_block == LinkedBlock::BLOCK_DATA_SIZE) {
				position_in_block = 0;
				current_pointer = block->next_block;
			}
		}
		return bytes_read;
	}
};

class LinkedBlockWriter {
private:
	FixedSizeAllocator &allocator;
	IndexPointer root_pointer;
	IndexPointer current_pointer;
	idx_t position_in_block;

public:
	LinkedBlockWriter(FixedSizeAllocator &allocator, IndexPointer root_pointer)
	    : allocator(allocator), root_pointer(root_pointer), current_pointer(root_pointer), position_in_block(0) {
	}

	void ClearCurrentBlock() {
		auto block = allocator.Get<LinkedBlock>(current_pointer, true);
		block->next_block.Clear();
		memset(block->data, 0, LinkedBlock::BLOCK_DATA_SIZE);
	}

	void Reset() {
		current_pointer = root_pointer;
		position_in_block = 0;
		ClearCurrentBlock();
	}

	void WriteData(const_data_ptr_t buffer, idx_t length) {
		idx_t bytes_written = 0;
		while (bytes_written < length) {
			auto block = allocator.Get<LinkedBlock>(current_pointer, true);
			auto block_data = block->data;
			auto data_to_write = std::min(length - bytes_written, LinkedBlock::BLOCK_DATA_SIZE - position_in_block);
			std::memcpy(block_data + position_in_block, buffer + bytes_written, data_to_write);

			bytes_written += data_to_write;
			position_in_block += data_to_write;

			if (position_in_block == LinkedBlock::BLOCK_DATA_SIZE) {
				position_in_block = 0;
				block->next_block = allocator.New();
				current_pointer = block->next_block;
				ClearCurrentBlock();
			}
		}
	}
};

//------------------------------------------------------------------------------
// HnswIndex
//------------------------------------------------------------------------------

HnswIndex::HnswIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
                     TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                     AttachedDatabase &db, const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                     idx_t estimated_cardinality)
    : VectorIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("HNSW indexes do not support unique or primary key constraints");
	}

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	linked_block_allocator = make_uniq<FixedSizeAllocator>(sizeof(LinkedBlock), block_manager);

	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);

	auto vector_size = ArrayType::GetSize(vector_type);
	auto vector_child_type = ArrayType::GetChildType(vector_type);

	auto scalar_kind = unum::usearch::scalar_kind_t::f32_k;
	auto scalar_kind_val = SCALAR_KIND_MAP.find(static_cast<uint8_t>(vector_child_type.id()));
	if (scalar_kind_val != SCALAR_KIND_MAP.end()) {
		scalar_kind = scalar_kind_val->second;
	}

	auto metric_kind = unum::usearch::metric_kind_t::l2sq_k;
	auto metric_kind_opt = options.find("metric");
	if (metric_kind_opt != options.end()) {
		auto metric_kind_val = METRIC_KIND_MAP.find(metric_kind_opt->second.GetValue<string>());
		if (metric_kind_val != METRIC_KIND_MAP.end()) {
			metric_kind = metric_kind_val->second;
		}
	}

	unum::usearch::metric_punned_t metric(vector_size, metric_kind, scalar_kind);
	unum::usearch::index_dense_config_t config = {};
	config.enable_key_lookups = false;

	auto ef_construction_opt = options.find("ef_construction");
	if (ef_construction_opt != options.end()) {
		config.expansion_add = ef_construction_opt->second.GetValue<int32_t>();
	}

	auto ef_search_opt = options.find("ef_search");
	if (ef_search_opt != options.end()) {
		config.expansion_search = ef_search_opt->second.GetValue<int32_t>();
	}

	auto m_opt = options.find("m");
	if (m_opt != options.end()) {
		config.connectivity = m_opt->second.GetValue<int32_t>();
		config.connectivity_base = config.connectivity * 2;
	}

	auto m0_opt = options.find("m0");
	if (m0_opt != options.end()) {
		config.connectivity_base = m0_opt->second.GetValue<int32_t>();
	}

	index = USearchIndexType::make(metric, config);

	auto lock = rwlock.GetExclusiveLock();
	if (info.IsValid()) {
		root_block_ptr.Set(info.root);
		D_ASSERT(info.allocator_infos.size() == 1);
		linked_block_allocator->Init(info.allocator_infos[0]);

		if (!info.allocator_infos[0].buffer_ids.empty()) {
			LinkedBlockReader reader(*linked_block_allocator, root_block_ptr);
			index.load_from_stream(
			    [&](void *data, size_t size) { return size == reader.ReadData(static_cast<data_ptr_t>(data), size); });
		}
	} else {
		index.reserve(MinValue(static_cast<idx_t>(32), estimated_cardinality));
	}
	index_size = index.size();
}

idx_t HnswIndex::GetVectorSize() const {
	return index.dimensions();
}

MetricKind HnswIndex::GetMetricKind() const {
	switch (index.metric().metric_kind()) {
	case unum::usearch::metric_kind_t::l2sq_k:
		return MetricKind::L2SQ;
	case unum::usearch::metric_kind_t::cos_k:
		return MetricKind::COSINE;
	case unum::usearch::metric_kind_t::ip_k:
		return MetricKind::IP;
	default:
		throw InternalException("Unknown metric kind");
	}
}

const case_insensitive_map_t<unum::usearch::metric_kind_t> HnswIndex::METRIC_KIND_MAP = {
    {"l2sq", unum::usearch::metric_kind_t::l2sq_k},
    {"cosine", unum::usearch::metric_kind_t::cos_k},
    {"ip", unum::usearch::metric_kind_t::ip_k},
};

const unordered_map<uint8_t, unum::usearch::scalar_kind_t> HnswIndex::SCALAR_KIND_MAP = {
    {static_cast<uint8_t>(LogicalTypeId::FLOAT), unum::usearch::scalar_kind_t::f32_k},
};

unique_ptr<HnswIndexStats> HnswIndex::GetStats() {
	auto lock = rwlock.GetExclusiveLock();
	auto result = make_uniq<HnswIndexStats>();

	result->max_level = index.max_level();
	result->count = index.size();
	result->capacity = index.capacity();
	result->approx_size = index.memory_usage();

	for (idx_t i = 0; i < index.max_level(); i++) {
		result->level_stats.push_back(index.stats(i));
	}
	return result;
}

//------------------------------------------------------------------------------
// Single-query scan
//------------------------------------------------------------------------------

struct HnswIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	unique_array<row_t> row_ids = nullptr;
};

unique_ptr<IndexScanState> HnswIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<HnswIndexScanState>();

	auto ef_search = index.expansion_search();
	Value hnsw_ef_search_opt;
	// vindex_ef_search is the new-world name; hnsw_ef_search remains as a
	// deprecated alias (see AGENTS.md §9 / src/common/index_pragmas.cpp).
	if (context.TryGetCurrentSetting("vindex_ef_search", hnsw_ef_search_opt) ||
	    context.TryGetCurrentSetting("hnsw_ef_search", hnsw_ef_search_opt)) {
		if (!hnsw_ef_search_opt.IsNull() && hnsw_ef_search_opt.type() == LogicalType::BIGINT) {
			auto val = hnsw_ef_search_opt.GetValue<int64_t>();
			if (val > 0) {
				ef_search = static_cast<idx_t>(val);
			}
		}
	}

	auto lock = rwlock.GetSharedLock();
	auto search_result = index.ef_search(query_vector, limit, ef_search);

	state->current_row = 0;
	state->total_rows = search_result.size();
	state->row_ids = make_uniq_array<row_t>(search_result.size());
	search_result.dump_to(state->row_ids.get());
	return std::move(state);
}

idx_t HnswIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<HnswIndexScanState>();

	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;

	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}
	return count;
}

//------------------------------------------------------------------------------
// Multi-query scan (join optimizer)
//------------------------------------------------------------------------------

struct MultiScanState final : IndexScanState {
	Vector vec;
	vector<row_t> row_ids;
	size_t ef_search;
	explicit MultiScanState(size_t ef_search_p) : vec(LogicalType::ROW_TYPE, nullptr), ef_search(ef_search_p) {
	}
};

unique_ptr<IndexScanState> HnswIndex::InitializeMultiScan(ClientContext &context) {
	auto ef_search = index.expansion_search();
	Value hnsw_ef_search_opt;
	if (context.TryGetCurrentSetting("vindex_ef_search", hnsw_ef_search_opt) ||
	    context.TryGetCurrentSetting("hnsw_ef_search", hnsw_ef_search_opt)) {
		if (!hnsw_ef_search_opt.IsNull() && hnsw_ef_search_opt.type() == LogicalType::BIGINT) {
			const auto val = hnsw_ef_search_opt.GetValue<int64_t>();
			if (val > 0) {
				ef_search = static_cast<idx_t>(val);
			}
		}
	}
	return make_uniq<MultiScanState>(ef_search);
}

idx_t HnswIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<MultiScanState>();

	USearchIndexType::search_result_t search_result;
	{
		auto lock = rwlock.GetSharedLock();
		search_result = index.ef_search(query_vector, limit, state.ef_search);
	}

	const auto offset = state.row_ids.size();
	state.row_ids.resize(state.row_ids.size() + search_result.size());
	search_result.dump_to(state.row_ids.data() + offset);
	return search_result.size();
}

const Vector &HnswIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<MultiScanState>();
	FlatVector::SetData(scan_state.vec, (data_ptr_t)scan_state.row_ids.data());
	return scan_state.vec;
}

void HnswIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<MultiScanState>();
	scan_state.row_ids.clear();
}

//------------------------------------------------------------------------------
// DuckDB BoundIndex contract
//------------------------------------------------------------------------------

void HnswIndex::CommitDrop(IndexLock &index_lock) {
	auto lock = rwlock.GetExclusiveLock();

	index.reset();
	index_size = 0;
	linked_block_allocator->Reset();
	root_block_ptr.Clear();
}

void HnswIndex::Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());

	is_dirty = true;

	auto count = input.size();
	input.Flatten();

	auto &vec_vec = input.data[0];
	auto &vec_child_vec = ArrayVector::GetEntry(vec_vec);
	auto array_size = ArrayType::GetSize(vec_vec.GetType());

	auto vec_child_data = FlatVector::GetData<float>(vec_child_vec);
	auto rowid_data = FlatVector::GetData<row_t>(row_ids);

	auto to_add_count = FlatVector::Validity(vec_vec).CountValid(count);

	bool needs_resize = false;
	{
		auto lock = rwlock.GetSharedLock();
		if (index_size.fetch_add(to_add_count) + to_add_count > index.capacity()) {
			needs_resize = true;
		}
	}

	if (needs_resize) {
		auto lock = rwlock.GetExclusiveLock();
		auto size = index_size.load();
		if (size > index.capacity()) {
			index.reserve(NextPowerOfTwo(size));
		}
	}

	{
		auto lock = rwlock.GetSharedLock();
		for (idx_t out_idx = 0; out_idx < count; out_idx++) {
			if (FlatVector::IsNull(vec_vec, out_idx)) {
				continue;
			}
			auto rowid = rowid_data[out_idx];
			auto result = index.add(rowid, vec_child_data + (out_idx * array_size), thread_idx);
			if (!result) {
				throw InternalException("Failed to add to the HNSW index: %s", result.error.what());
			}
		}
	}
}

void HnswIndex::Compact() {
	is_dirty = true;

	auto lock = rwlock.GetExclusiveLock();
	auto result = index.compact();
	if (!result) {
		throw InternalException("Failed to compact the HNSW index: %s", result.error.what());
	}
	index_size = index.size();
}

void HnswIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	is_dirty = true;

	auto count = input.size();
	rowid_vec.Flatten(count);
	auto row_id_data = FlatVector::GetData<row_t>(rowid_vec);

	auto _lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < input.size(); i++) {
		auto result = index.remove(row_id_data[i]);
		(void)result;
	}
	index_size = index.size();
}

ErrorData HnswIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	Construct(input, rowid_vec, unum::usearch::index_dense_t::any_thread());
	return ErrorData {};
}

ErrorData HnswIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(appended_data, expression_result);
	Construct(expression_result, row_identifiers, unum::usearch::index_dense_t::any_thread());
	return ErrorData {};
}

void HnswIndex::PersistToDisk() {
	auto lock = rwlock.GetExclusiveLock();

	if (!is_dirty) {
		return;
	}

	if (root_block_ptr.Get() == 0) {
		root_block_ptr = linked_block_allocator->New();
	}

	LinkedBlockWriter writer(*linked_block_allocator, root_block_ptr);
	writer.Reset();
	index.save_to_stream([&](const void *data, size_t size) {
		writer.WriteData(static_cast<const_data_ptr_t>(data), size);
		return true;
	});

	is_dirty = false;
}

IndexStorageInfo HnswIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr.Get();

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	linked_block_allocator->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
	info.allocator_infos.push_back(linked_block_allocator->GetInfo());

	return info;
}

IndexStorageInfo HnswIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr.Get();
	info.buffers.push_back(linked_block_allocator->InitSerializationToWAL());
	info.allocator_infos.push_back(linked_block_allocator->GetInfo());

	return info;
}

idx_t HnswIndex::GetInMemorySize(IndexLock &state) {
	return index.memory_usage();
}

bool HnswIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("HnswIndex::MergeIndexes() not implemented");
}

void HnswIndex::Vacuum(IndexLock &state) {
}

void HnswIndex::Verify(IndexLock &state) {
	throw NotImplementedException("HnswIndex::Verify() not implemented");
}

string HnswIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("HnswIndex::ToString() not implemented");
}

void HnswIndex::VerifyAllocations(IndexLock &state) {
	throw NotImplementedException("HnswIndex::VerifyAllocations() not implemented");
}

void HnswIndex::VerifyBuffers(IndexLock &lock) {
	linked_block_allocator->VerifyBuffers();
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

	// Persistence opt-in. AGENTS.md §9: preferred name is
	// `vindex_enable_experimental_persistence`; `hnsw_*` is kept as a
	// deprecated alias for at least one minor release.
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
	if (!db.config.GetOptionByName("hnsw_ef_search")) {
		db.config.AddExtensionOption("hnsw_ef_search", "deprecated alias of vindex_ef_search", LogicalType::BIGINT);
	}

	db.config.GetIndexTypes().RegisterIndexType(index_type);

	VectorIndexRegistry::Instance().RegisterTypeName(HnswIndex::TYPE_NAME);
}

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
