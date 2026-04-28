#pragma once

#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/table/table_scan.hpp"

namespace duckdb {
class Index;

namespace vindex {

// Bind data carried from the optimizer rule into the replaced table scan.
// Holds the materialised query vector + top-K limit for a single ANN query.
// Algorithm-agnostic: dispatches through the VectorIndex virtual surface
// (InitializeScan/Scan), so HNSW / IVF / DiskANN all share this one path.
struct VectorIndexScanBindData final : public TableScanBindData {
	VectorIndexScanBindData(TableCatalogEntry &table, Index &index, idx_t limit, unsafe_unique_array<float> query)
	    : TableScanBindData(table), index(index), limit(limit), query(std::move(query)) {
	}

	Index &index;
	idx_t limit;
	unsafe_unique_array<float> query;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<VectorIndexScanBindData>();
		return &other.table == &table;
	}
};

struct VectorIndexScanFunction {
	// Registered under the name `vindex_index_scan`. The optimizer rewrites
	// plans to use this function for every VectorIndex subclass.
	static TableFunction GetFunction();
};

} // namespace vindex
} // namespace duckdb
