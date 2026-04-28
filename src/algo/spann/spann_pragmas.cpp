#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/function/pragma_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "algo/spann/spann_index.hpp"

namespace duckdb {
namespace vindex {
namespace spann {

//-------------------------------------------------------------------------
// pragma_vindex_spann_index_info() — one row per SPANN index.
//-------------------------------------------------------------------------

static unique_ptr<FunctionData> SpannIndexInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("catalog_name");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("index_name");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("metric");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("dimensions");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("count");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("nlist");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("nprobe");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("replica_count");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("closure_factor");
	return_types.emplace_back(LogicalType::FLOAT);
	names.emplace_back("entries");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("approx_memory_usage");
	return_types.emplace_back(LogicalType::BIGINT);
	return nullptr;
}

struct SpannIndexInfoGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	vector<reference<IndexCatalogEntry>> entries;
};

static unique_ptr<GlobalTableFunctionState> SpannIndexInfoInitGlobal(ClientContext &context,
                                                                     TableFunctionInitInput &input) {
	auto result = make_uniq<SpannIndexInfoGlobalState>();
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type == SpannIndex::TYPE_NAME) {
				result->entries.push_back(index_entry);
			}
		});
	}
	return std::move(result);
}

static void SpannIndexInfoExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<SpannIndexInfoGlobalState>();
	if (data.offset >= data.entries.size()) {
		return;
	}

	idx_t row = 0;
	while (data.offset < data.entries.size() && row < STANDARD_VECTOR_SIZE) {
		auto &index_entry = data.entries[data.offset++].get();
		auto &table_entry = index_entry.schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry.GetSchemaName(),
		                                                                           index_entry.GetTableName());
		auto &storage = table_entry.GetStorage();
		SpannIndex *spann_index = nullptr;

		auto &table_info = *storage.GetDataTableInfo();
		table_info.BindIndexes(context, SpannIndex::TYPE_NAME);
		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || SpannIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<SpannIndex>();
			if (cast_index.name == index_entry.name) {
				spann_index = &cast_index;
				break;
			}
		}
		if (!spann_index) {
			throw BinderException("Index %s not found", index_entry.name);
		}

		idx_t col = 0;
		output.data[col++].SetValue(row, Value(index_entry.catalog.GetName()));
		output.data[col++].SetValue(row, Value(index_entry.schema.name));
		output.data[col++].SetValue(row, Value(index_entry.name));
		output.data[col++].SetValue(row, Value(table_entry.name));
		output.data[col++].SetValue(row, Value(MetricName(spann_index->GetMetricKind())));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->GetVectorSize()));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->Count()));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->NumCentroids()));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->Nprobe()));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->ReplicaCount()));
		output.data[col++].SetValue(row, Value::FLOAT(spann_index->ClosureFactor()));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->Entries()));
		output.data[col++].SetValue(row, Value::BIGINT(spann_index->ApproxMemory()));
		row++;
	}
	output.SetCardinality(row);
}

void RegisterPragmas(ExtensionLoader &loader) {
	TableFunction info_function("pragma_vindex_spann_index_info", {}, SpannIndexInfoExecute, SpannIndexInfoBind,
	                            SpannIndexInfoInitGlobal);
	loader.RegisterFunction(info_function);
}

} // namespace spann
} // namespace vindex
} // namespace duckdb
