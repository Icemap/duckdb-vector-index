#include "algo/hnsw/hnsw_module.hpp"

#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "algo/hnsw/hnsw_index_scan.hpp"

namespace duckdb {
namespace vindex {
namespace hnsw {

// Defined in hnsw_index.cpp — registers the HNSW IndexType + extension
// options (and adds the TYPE_NAME to VectorIndexRegistry).
void RegisterIndex(DatabaseInstance &db);
void RegisterPragmas(ExtensionLoader &loader);

void Register(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	RegisterIndex(db);
	loader.RegisterFunction(HnswIndexScanFunction::GetFunction());
	RegisterPragmas(loader);

	// Optimizers + macros are algorithm-agnostic; registered once from
	// src/common/ rather than per-algorithm.
}

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
