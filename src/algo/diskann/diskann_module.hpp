#pragma once

namespace duckdb {
class ExtensionLoader;
} // namespace duckdb

namespace duckdb {
namespace vindex {
namespace diskann {

// Register the DiskANN index type + its pragma hooks. Called from
// vindex::RegisterBuiltInAlgorithms().
void Register(ExtensionLoader &loader);

} // namespace diskann
} // namespace vindex
} // namespace duckdb
