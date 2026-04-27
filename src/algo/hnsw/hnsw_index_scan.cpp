#include "algo/hnsw/hnsw_index_scan.hpp"

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

#include "algo/hnsw/hnsw_index.hpp"

namespace duckdb {
namespace vindex {
namespace hnsw {

static BindInfo HnswIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HnswIndexScanBindData>();
	return BindInfo(bind_data.table);
}

//-------------------------------------------------------------------------
// Global State
//-------------------------------------------------------------------------
struct HnswIndexScanGlobalState : public GlobalTableFunctionState {
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;

	unique_ptr<IndexScanState> index_state;
	Vector row_ids = Vector(LogicalType::ROW_TYPE);

	DataChunk all_columns;
	vector<idx_t> projection_ids;
};

static unique_ptr<GlobalTableFunctionState> HnswIndexScanInitGlobal(ClientContext &context,
                                                                    TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<HnswIndexScanBindData>();

	auto result = make_uniq<HnswIndexScanGlobalState>();

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

	result->index_state =
	    bind_data.index.Cast<HnswIndex>().InitializeScan(bind_data.query.get(), bind_data.limit, context);

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
static void HnswIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<HnswIndexScanBindData>();
	auto &state = data_p.global_state->Cast<HnswIndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

	auto row_count = bind_data.index.Cast<HnswIndex>().Scan(*state.index_state, state.row_ids);
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
static unique_ptr<BaseStatistics> HnswIndexScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                          column_t column_id) {
	auto &bind_data = bind_data_p->Cast<HnswIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

static void HnswIndexScanDependency(LogicalDependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HnswIndexScanBindData>();
	entries.AddDependency(bind_data.table);
}

static unique_ptr<NodeStatistics> HnswIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HnswIndexScanBindData>();
	return make_uniq<NodeStatistics>(bind_data.limit, bind_data.limit);
}

static InsertionOrderPreservingMap<string> HnswIndexScanToString(TableFunctionToStringInput &input) {
	D_ASSERT(input.bind_data);
	InsertionOrderPreservingMap<string> result;
	auto &bind_data = input.bind_data->Cast<HnswIndexScanBindData>();
	result["Table"] = bind_data.table.name;
	result["HNSW Index"] = bind_data.index.GetIndexName();
	return result;
}

//-------------------------------------------------------------------------
// Get Function
//-------------------------------------------------------------------------
TableFunction HnswIndexScanFunction::GetFunction() {
	TableFunction func("hnsw_index_scan", {}, HnswIndexScanExecute);
	func.init_local = nullptr;
	func.init_global = HnswIndexScanInitGlobal;
	func.statistics = HnswIndexScanStatistics;
	func.dependency = HnswIndexScanDependency;
	func.cardinality = HnswIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = HnswIndexScanToString;
	func.table_scan_progress = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = HnswIndexScanBindInfo;
	return func;
}

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
