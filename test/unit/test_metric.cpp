// Unit tests for src/common/metric.cpp — parsing, reverse mapping, and the
// SQL distance-function names each MetricKind rewrites to.

#include <catch2/catch_test_macros.hpp>

#include "duckdb/common/exception/binder_exception.hpp"

#include "vindex/metric.hpp"

using duckdb::vindex::DistanceFunctionNames;
using duckdb::vindex::MetricKind;
using duckdb::vindex::MetricName;
using duckdb::vindex::ParseMetric;

TEST_CASE("ParseMetric accepts canonical names", "[metric][unit]") {
	REQUIRE(ParseMetric("l2sq") == MetricKind::L2SQ);
	REQUIRE(ParseMetric("cosine") == MetricKind::COSINE);
	REQUIRE(ParseMetric("ip") == MetricKind::IP);
}

TEST_CASE("ParseMetric is case-insensitive", "[metric][unit]") {
	REQUIRE(ParseMetric("L2SQ") == MetricKind::L2SQ);
	REQUIRE(ParseMetric("Cosine") == MetricKind::COSINE);
	REQUIRE(ParseMetric("IP") == MetricKind::IP);
}

TEST_CASE("ParseMetric rejects unknown names", "[metric][unit]") {
	REQUIRE_THROWS_AS(ParseMetric("euclidean"), duckdb::BinderException);
	REQUIRE_THROWS_AS(ParseMetric(""), duckdb::BinderException);
	REQUIRE_THROWS_AS(ParseMetric("l2"), duckdb::BinderException);
}

TEST_CASE("MetricName round-trips with ParseMetric", "[metric][unit]") {
	for (auto kind : {MetricKind::L2SQ, MetricKind::COSINE, MetricKind::IP}) {
		REQUIRE(ParseMetric(MetricName(kind)) == kind);
	}
}

TEST_CASE("DistanceFunctionNames matches DuckDB scalar-function names", "[metric][unit]") {
	// These strings are what the scan/topk optimizers grep for when rewriting
	// ORDER BY array_distance(...) into an index scan. If they drift out of
	// sync with DuckDB's catalog, the optimizer silently stops firing.
	auto l2 = DistanceFunctionNames(MetricKind::L2SQ);
	REQUIRE(l2 == std::vector<duckdb::string> {"array_distance", "<->"});

	auto cos = DistanceFunctionNames(MetricKind::COSINE);
	REQUIRE(cos == std::vector<duckdb::string> {"array_cosine_distance", "<=>"});

	auto ip = DistanceFunctionNames(MetricKind::IP);
	REQUIRE(ip == std::vector<duckdb::string> {"array_negative_inner_product", "<#>"});
}
