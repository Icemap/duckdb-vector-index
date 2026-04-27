#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/string.hpp"

namespace duckdb {
namespace vindex {

// Supported distance metrics. Names align with DuckDB's array_distance family
// so the optimizer can rewrite ORDER BY array_distance(...) into an index scan
// for any algorithm that reports the matching MetricKind.
enum class MetricKind : uint8_t {
	L2SQ = 0,   // array_distance
	COSINE = 1, // array_cosine_distance
	IP = 2,     // array_negative_inner_product
};

// Parse the `metric = '...'` option string. Throws BinderException on unknown.
MetricKind ParseMetric(const string &name);

// Reverse mapping, used by VectorIndex::GetMetric().
const char *MetricName(MetricKind kind);

// The DuckDB scalar function names that correspond to each metric. Used by
// common/optimize_scan.cpp to match ORDER BY expressions.
vector<string> DistanceFunctionNames(MetricKind kind);

} // namespace vindex
} // namespace duckdb
