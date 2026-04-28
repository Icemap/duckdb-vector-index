#include "vindex/vector_index_scan.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"

#include "vindex/vector_index.hpp"
#include "vindex/vector_index_registry.hpp"

namespace duckdb {
namespace vindex {

static BindInfo VectorIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<VectorIndexScanBindData>();
	return BindInfo(bind_data.table);
}

//-------------------------------------------------------------------------
// Global State
//-------------------------------------------------------------------------
struct VectorIndexScanGlobalState : public GlobalTableFunctionState {
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;

	unique_ptr<IndexScanState> index_state;
	Vector row_ids = Vector(LogicalType::ROW_TYPE);

	DataChunk all_columns;
	vector<idx_t> projection_ids;
};

static unique_ptr<GlobalTableFunctionState> VectorIndexScanInitGlobal(ClientContext &context,
                                                                     TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<VectorIndexScanBindData>();

	auto result = make_uniq<VectorIndexScanGlobalState>();

	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	result->column_ids.reserve(input.column_ids.size());

	for (auto &id : input.column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = bind_data.table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->column_ids.emplace_back(col_id);
	}

	result->local_storage_state.Initialize(result->column_ids, context, input.filters);
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	auto *vi = VectorIndexRegistry::TryCast(bind_data.index.Cast<BoundIndex>());
	if (!vi) {
		throw InternalException("vindex_index_scan invoked on a non-VectorIndex");
	}
	result->index_state = vi->InitializeScan(bind_data.query.get(), bind_data.limit, context);

	if (!input.CanRemoveFilterColumns()) {
		return std::move(result);
	}

	result->projection_ids = input.projection_ids;

	auto &duck_table = bind_data.table.Cast<DuckTableEntry>();
	const auto &columns = duck_table.GetColumns();
	vector<LogicalType> scanned_types;
	for (const auto &col_idx : input.column_indexes) {
		if (col_idx.IsRowIdColumn()) {
			scanned_types.emplace_back(LogicalType::ROW_TYPE);
		} else {
			scanned_types.push_back(columns.GetColumn(col_idx.ToLogical()).Type());
		}
	}
	result->all_columns.Initialize(context, scanned_types);

	return std::move(result);
}

//-------------------------------------------------------------------------
// Execute
//-------------------------------------------------------------------------
static void VectorIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<VectorIndexScanBindData>();
	auto &state = data_p.global_state->Cast<VectorIndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

	auto *vi = VectorIndexRegistry::TryCast(bind_data.index.Cast<BoundIndex>());
	if (!vi) {
		throw InternalException("vindex_index_scan invoked on a non-VectorIndex");
	}
	auto row_count = vi->Scan(*state.index_state, state.row_ids);
	if (row_count == 0) {
		output.SetCardinality(0);
		return;
	}

	if (state.projection_ids.empty()) {
		bind_data.table.GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids, row_count,
		                                   state.fetch_state);
		return;
	}

	state.all_columns.Reset();
	bind_data.table.GetStorage().Fetch(transaction, state.all_columns, state.column_ids, state.row_ids, row_count,
	                                   state.fetch_state);
	output.ReferenceColumns(state.all_columns, state.projection_ids);
}

//-------------------------------------------------------------------------
// Statistics / Dependency / Cardinality / ToString
//-------------------------------------------------------------------------
static unique_ptr<BaseStatistics> VectorIndexScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                            column_t column_id) {
	auto &bind_data = bind_data_p->Cast<VectorIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

static void VectorIndexScanDependency(LogicalDependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<VectorIndexScanBindData>();
	entries.AddDependency(bind_data.table);
}

static unique_ptr<NodeStatistics> VectorIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<VectorIndexScanBindData>();
	return make_uniq<NodeStatistics>(bind_data.limit, bind_data.limit);
}

static InsertionOrderPreservingMap<string> VectorIndexScanToString(TableFunctionToStringInput &input) {
	D_ASSERT(input.bind_data);
	InsertionOrderPreservingMap<string> result;
	auto &bind_data = input.bind_data->Cast<VectorIndexScanBindData>();
	result["Table"] = bind_data.table.name;
	// The EXPLAIN string still carries the algorithm name so that user-
	// facing plans read `HNSW_INDEX_SCAN` / `IVF_INDEX_SCAN` / ... even
	// though there is a single underlying table function.
	const auto type_name = bind_data.index.GetIndexType();
	result[type_name + " Index"] = bind_data.index.GetIndexName();
	return result;
}

//-------------------------------------------------------------------------
// Get Function
//-------------------------------------------------------------------------
TableFunction VectorIndexScanFunction::GetFunction() {
	TableFunction func("vindex_index_scan", {}, VectorIndexScanExecute);
	func.init_local = nullptr;
	func.init_global = VectorIndexScanInitGlobal;
	func.statistics = VectorIndexScanStatistics;
	func.dependency = VectorIndexScanDependency;
	func.cardinality = VectorIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = VectorIndexScanToString;
	func.table_scan_progress = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = VectorIndexScanBindInfo;
	return func;
}

} // namespace vindex
} // namespace duckdb
