#pragma once

namespace duckdb {
class ExtensionLoader;
} // namespace duckdb

namespace duckdb {
namespace vindex {
namespace spann {

// Register the SPANN index type + its pragma hooks. Called from
// vindex::RegisterBuiltInAlgorithms().
void Register(ExtensionLoader &loader);

} // namespace spann
} // namespace vindex
} // namespace duckdb
