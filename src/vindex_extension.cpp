#include "vindex_extension.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

#include "vindex/vector_index_registry.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	// Register every algorithm's index type, plus the shared optimizers,
	// pragmas and macros. All of them live under src/common/ and src/algo/.
	vindex::RegisterBuiltInAlgorithms(loader);
}

void VindexExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string VindexExtension::Name() {
	return "vindex";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(vindex, loader) {
	duckdb::LoadInternal(loader);
}
}
