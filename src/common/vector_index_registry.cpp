#include "vindex/vector_index_registry.hpp"

#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/storage/index.hpp"

#include "vindex/vector_index.hpp"
#include "vindex/vector_index_scan.hpp"

// Forward declarations of per-algorithm registration entrypoints. Each
// algorithm's module.hpp exposes a `Register(ExtensionLoader&)` symbol.
#include "algo/diskann/diskann_module.hpp"
#include "algo/hnsw/hnsw_module.hpp"
#include "algo/ivf/ivf_module.hpp"
#include "algo/spann/spann_module.hpp"

namespace duckdb {
namespace vindex {

VectorIndexRegistry &VectorIndexRegistry::Instance() {
	static VectorIndexRegistry instance;
	return instance;
}

void VectorIndexRegistry::RegisterTypeName(const string &name) {
	type_names_.insert(name);
}

VectorIndex *VectorIndexRegistry::TryCast(BoundIndex &index) {
	// Only downcast if the index's type name was registered by one of our
	// algorithms. This keeps us from accidentally claiming third-party
	// indexes that happen to inherit from BoundIndex.
	auto &names = Instance().type_names_;
	if (names.find(index.GetIndexType()) == names.end()) {
		return nullptr;
	}
	return dynamic_cast<VectorIndex *>(&index);
}

// Optimizer entrypoints live in src/common/optimize_*.cpp. They are generic
// (registry-driven) rather than per-algorithm.
void RegisterExprOptimizer(DatabaseInstance &db);
void RegisterScanOptimizer(DatabaseInstance &db);
void RegisterTopKOptimizer(DatabaseInstance &db);
void RegisterJoinOptimizer(DatabaseInstance &db);
void RegisterMacros(ExtensionLoader &loader);
void RegisterCompactPragma(ExtensionLoader &loader);

void RegisterBuiltInAlgorithms(ExtensionLoader &loader) {
	// 1) Algorithm-specific index types + pragmas.
	hnsw::Register(loader);
	ivf::Register(loader);
	diskann::Register(loader);
	spann::Register(loader);
	RegisterCompactPragma(loader);

	// 2) Shared optimizers — registered once, dispatch via VectorIndexRegistry.
	auto &db = loader.GetDatabaseInstance();
	RegisterExprOptimizer(db);
	RegisterScanOptimizer(db);
	RegisterTopKOptimizer(db);
	RegisterJoinOptimizer(db);

	// 3) Shared table function — a single generic vindex_index_scan handles
	//    every VectorIndex subclass via virtual dispatch. Registering it
	//    here (rather than per-algorithm) is what keeps new algorithms from
	//    having to duplicate the scan plumbing.
	loader.RegisterFunction(VectorIndexScanFunction::GetFunction());

	// 4) Shared SQL macros.
	RegisterMacros(loader);
}

} // namespace vindex
} // namespace duckdb
