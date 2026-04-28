#include "algo/spann/spann_module.hpp"

#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace vindex {
namespace spann {

// Defined in spann_index.cpp / spann_pragmas.cpp.
void RegisterIndex(DatabaseInstance &db);
void RegisterPragmas(ExtensionLoader &loader);

void Register(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	RegisterIndex(db);
	RegisterPragmas(loader);

	// Optimizers, macros, and the generic vindex_index_scan table function
	// are algorithm-agnostic; registered once from src/common/.
}

} // namespace spann
} // namespace vindex
} // namespace duckdb
