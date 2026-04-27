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

#include "algo/hnsw/hnsw_index.hpp"

namespace duckdb {
namespace vindex {
namespace hnsw {

//-------------------------------------------------------------------------
// pragma_hnsw_index_info() — verbatim port, returns one row per HNSW index.
//-------------------------------------------------------------------------

static unique_ptr<FunctionData> HnswIndexInfoBind(ClientContext &context, TableFunctionBindInput &input,
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
	names.emplace_back("capacity");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("approx_memory_usage");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("levels");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("levels_stats");
	return_types.emplace_back(LogicalType::LIST(LogicalType::STRUCT({{"nodes", LogicalType::BIGINT},
	                                                                 {"edges", LogicalType::BIGINT},
	                                                                 {"max_edges", LogicalType::BIGINT},
	                                                                 {"allocated_bytes", LogicalType::BIGINT}})));
	return nullptr;
}

struct HnswIndexInfoGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	vector<reference<IndexCatalogEntry>> entries;
};

static unique_ptr<GlobalTableFunctionState> HnswIndexInfoInitGlobal(ClientContext &context,
                                                                    TableFunctionInitInput &input) {
	auto result = make_uniq<HnswIndexInfoGlobalState>();
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type == HnswIndex::TYPE_NAME) {
				result->entries.push_back(index_entry);
			}
		});
	}
	return std::move(result);
}

static const char *MetricName(unum::usearch::metric_kind_t kind) {
	switch (kind) {
	case unum::usearch::metric_kind_t::l2sq_k:
		return "l2sq";
	case unum::usearch::metric_kind_t::cos_k:
		return "cosine";
	case unum::usearch::metric_kind_t::ip_k:
		return "ip";
	default:
		return "unknown";
	}
}

static void HnswIndexInfoExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<HnswIndexInfoGlobalState>();
	if (data.offset >= data.entries.size()) {
		return;
	}

	idx_t row = 0;
	while (data.offset < data.entries.size() && row < STANDARD_VECTOR_SIZE) {
		auto &index_entry = data.entries[data.offset++].get();
		auto &table_entry = index_entry.schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry.GetSchemaName(),
		                                                                           index_entry.GetTableName());
		auto &storage = table_entry.GetStorage();
		HnswIndex *hnsw_index = nullptr;

		auto &table_info = *storage.GetDataTableInfo();
		table_info.BindIndexes(context, HnswIndex::TYPE_NAME);
		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound() || HnswIndex::TYPE_NAME != index.GetIndexType()) {
				continue;
			}
			auto &cast_index = index.Cast<HnswIndex>();
			if (cast_index.name == index_entry.name) {
				hnsw_index = &cast_index;
				break;
			}
		}

		if (!hnsw_index) {
			throw BinderException("Index %s not found", index_entry.name);
		}

		idx_t col = 0;
		output.data[col++].SetValue(row, Value(index_entry.catalog.GetName()));
		output.data[col++].SetValue(row, Value(index_entry.schema.name));
		output.data[col++].SetValue(row, Value(index_entry.name));
		output.data[col++].SetValue(row, Value(table_entry.name));

		auto stats = hnsw_index->GetStats();
		output.data[col++].SetValue(row, Value(MetricName(hnsw_index->index.metric().metric_kind())));
		output.data[col++].SetValue(row, Value::BIGINT(hnsw_index->GetVectorSize()));
		output.data[col++].SetValue(row, Value::BIGINT(stats->count));
		output.data[col++].SetValue(row, Value::BIGINT(stats->capacity));
		output.data[col++].SetValue(row, Value::BIGINT(stats->approx_size));
		output.data[col++].SetValue(row, Value::BIGINT(stats->max_level));

		vector<Value> level_stats;
		for (auto &stat : stats->level_stats) {
			level_stats.push_back(Value::STRUCT({{"nodes", Value::BIGINT(stat.nodes)},
			                                     {"edges", Value::BIGINT(stat.edges)},
			                                     {"max_edges", Value::BIGINT(stat.max_edges)},
			                                     {"allocated_bytes", Value::BIGINT(stat.allocated_bytes)}}));
		}
		auto level_stat_value = Value::LIST(LogicalType::STRUCT({{{"nodes", LogicalType::BIGINT},
		                                                          {"edges", LogicalType::BIGINT},
		                                                          {"max_edges", LogicalType::BIGINT},
		                                                          {"allocated_bytes", LogicalType::BIGINT}}}),
		                                    level_stats);
		output.data[col++].SetValue(row, level_stat_value);
		row++;
	}
	output.SetCardinality(row);
}

//-------------------------------------------------------------------------
// Compact PRAGMA
//-------------------------------------------------------------------------

static void CompactIndexPragma(ClientContext &context, const FunctionParameters &parameters) {
	if (parameters.values.size() != 1) {
		throw BinderException("Expected one argument for hnsw_compact_index");
	}
	auto &param = parameters.values[0];
	if (param.type() != LogicalType::VARCHAR) {
		throw BinderException("Expected a string argument for hnsw_compact_index");
	}
	auto index_name = param.GetValue<string>();

	auto qname = QualifiedName::Parse(index_name);
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto &index_entry = Catalog::GetEntry(context, CatalogType::INDEX_ENTRY, qname.catalog, qname.schema, qname.name)
	                        .Cast<IndexCatalogEntry>();
	auto &table_entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, qname.catalog, index_entry.GetSchemaName(),
	                                      index_entry.GetTableName())
	                        .Cast<TableCatalogEntry>();

	auto &storage = table_entry.GetStorage();
	bool found_index = false;
	auto &table_info = *storage.GetDataTableInfo();
	table_info.BindIndexes(context, HnswIndex::TYPE_NAME);
	for (auto &index : table_info.GetIndexes().Indexes()) {
		if (!index.IsBound() || HnswIndex::TYPE_NAME != index.GetIndexType()) {
			continue;
		}
		auto &cast_index = index.Cast<HnswIndex>();
		if (cast_index.name == index_entry.name) {
			cast_index.Compact();
			found_index = true;
			break;
		}
	}

	if (!found_index) {
		throw BinderException("Index %s not found", index_name);
	}
}

//-------------------------------------------------------------------------
// Register — both `vindex_*` (preferred) and `hnsw_*` (deprecated alias).
//-------------------------------------------------------------------------
void RegisterPragmas(ExtensionLoader &loader) {
	// Preferred names.
	loader.RegisterFunction(
	    PragmaFunction::PragmaCall("vindex_compact_index", CompactIndexPragma, {LogicalType::VARCHAR}));
	TableFunction info_function("pragma_vindex_hnsw_index_info", {}, HnswIndexInfoExecute, HnswIndexInfoBind,
	                            HnswIndexInfoInitGlobal);
	loader.RegisterFunction(info_function);

	// Legacy aliases — one minor release of overlap per AGENTS.md §9.
	loader.RegisterFunction(
	    PragmaFunction::PragmaCall("hnsw_compact_index", CompactIndexPragma, {LogicalType::VARCHAR}));
	TableFunction legacy_info_function("pragma_hnsw_index_info", {}, HnswIndexInfoExecute, HnswIndexInfoBind,
	                                   HnswIndexInfoInitGlobal);
	loader.RegisterFunction(legacy_info_function);
}

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
