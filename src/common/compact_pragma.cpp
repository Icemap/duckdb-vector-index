#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/function/pragma_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "vindex/vector_index.hpp"
#include "vindex/vector_index_registry.hpp"

namespace duckdb {
namespace vindex {

// `PRAGMA vindex_compact_index('<idx>')` — algorithm-agnostic: looks up
// the index by name and dispatches to VectorIndex::Compact() via the
// registry. Each algorithm decides what "compact" does internally.
// `PRAGMA hnsw_compact_index('<idx>')` remains as a deprecated alias
// for one release per AGENTS §4.2.
static void CompactIndexPragma(ClientContext &context, const FunctionParameters &parameters) {
	if (parameters.values.size() != 1) {
		throw BinderException("Expected one argument for vindex_compact_index");
	}
	auto &param = parameters.values[0];
	if (param.type() != LogicalType::VARCHAR) {
		throw BinderException("Expected a string argument for vindex_compact_index");
	}
	const auto index_name = param.GetValue<string>();

	auto qname = QualifiedName::Parse(index_name);
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto &index_entry = Catalog::GetEntry(context, CatalogType::INDEX_ENTRY, qname.catalog, qname.schema, qname.name)
	                        .Cast<IndexCatalogEntry>();
	auto &table_entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, qname.catalog, index_entry.GetSchemaName(),
	                                      index_entry.GetTableName())
	                        .Cast<TableCatalogEntry>();

	auto &storage = table_entry.GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	table_info.BindIndexes(context, index_entry.index_type.c_str());

	auto bound = table_info.GetIndexes().Find(index_entry.name);
	if (!bound) {
		throw BinderException("Index %s not found", index_name);
	}

	auto *vindex = VectorIndexRegistry::TryCast(*bound);
	if (!vindex) {
		throw BinderException("vindex_compact_index: '%s' is not a vector index", index_name);
	}
	vindex->Compact();
}

void RegisterCompactPragma(ExtensionLoader &loader) {
	loader.RegisterFunction(
	    PragmaFunction::PragmaCall("vindex_compact_index", CompactIndexPragma, {LogicalType::VARCHAR}));
	// Deprecated alias — `duckdb-vss` exposed `hnsw_compact_index(...)`.
	loader.RegisterFunction(
	    PragmaFunction::PragmaCall("hnsw_compact_index", CompactIndexPragma, {LogicalType::VARCHAR}));
}

} // namespace vindex
} // namespace duckdb
