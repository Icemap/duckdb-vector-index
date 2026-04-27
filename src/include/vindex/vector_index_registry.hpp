#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/unordered_set.hpp"

#include "vindex/vector_index.hpp"

namespace duckdb {
class DatabaseInstance;
class ExtensionLoader;
class BoundIndex;
} // namespace duckdb

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// VectorIndexRegistry — single source of truth for which algorithms exist.
//
// Each algorithm has a module (hnsw::Register(), ivf::Register(), ...) that:
//   1. Registers a DuckDB IndexType (so CREATE INDEX ... USING <NAME> works).
//   2. Adds its TYPE_NAME to this registry, so the common optimizers can
//      enumerate all algorithm type names instead of hard-coding one.
//
// The registry is a process-global singleton. All member access happens from
// the optimizer / bind paths, which DuckDB serialises inside a ClientContext
// (no concurrent writes once the extension finished loading).
// ---------------------------------------------------------------------------

class VectorIndexRegistry {
public:
	static VectorIndexRegistry &Instance();

	void RegisterTypeName(const string &name);

	const unordered_set<string> &TypeNames() const {
		return type_names_;
	}

	// Convenience cast used by the common optimizers: returns non-null only if
	// `index` is one of our registered algorithm types AND is a VectorIndex.
	static VectorIndex *TryCast(BoundIndex &index);

private:
	unordered_set<string> type_names_;
};

// Called once from VindexExtension::Load. Populates both the registry and
// the DuckDB IndexTypeSet / optimizer extensions for every built-in algorithm.
void RegisterBuiltInAlgorithms(ExtensionLoader &loader);

} // namespace vindex
} // namespace duckdb
