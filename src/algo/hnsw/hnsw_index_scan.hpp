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
namespace hnsw {

// Bind data carried from the optimizer rule into the replaced table scan.
// Holds the materialised query vector + top-K limit for a single ANN query.
struct HnswIndexScanBindData final : public TableScanBindData {
	HnswIndexScanBindData(TableCatalogEntry &table, Index &index, idx_t limit, unsafe_unique_array<float> query)
	    : TableScanBindData(table), index(index), limit(limit), query(std::move(query)) {
	}

	Index &index;
	idx_t limit;
	unsafe_unique_array<float> query;

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<HnswIndexScanBindData>();
		return &other.table == &table;
	}
};

struct HnswIndexScanFunction {
	// Registered under the name `hnsw_index_scan` for compatibility with vss;
	// the optimizer rewrites plans to use this function. A future generic
	// name (e.g. `vindex_index_scan`) can be layered in as an alias.
	static TableFunction GetFunction();
};

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
