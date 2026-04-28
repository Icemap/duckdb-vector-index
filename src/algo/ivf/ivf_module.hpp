#pragma once

namespace duckdb {
class ExtensionLoader;
} // namespace duckdb

namespace duckdb {
namespace vindex {
namespace ivf {

// Register the IVF index type + its pragma hooks. Called from
// vindex::RegisterBuiltInAlgorithms().
void Register(ExtensionLoader &loader);

} // namespace ivf
} // namespace vindex
} // namespace duckdb
