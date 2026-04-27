#pragma once

namespace duckdb {
class ExtensionLoader;
} // namespace duckdb

namespace duckdb {
namespace vindex {
namespace hnsw {

// Register the HNSW index type + its optimizer/pragma/macro hooks. Called
// from vindex::RegisterBuiltInAlgorithms().
void Register(ExtensionLoader &loader);

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
