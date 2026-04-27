// Unit tests for FlatQuantizer (src/quant/flat/flat_quantizer.cpp).
// FlatQuantizer is the identity codec: EstimateDistance must reproduce the
// exact metric values DuckDB's array_distance / array_cosine_distance /
// array_negative_inner_product functions return, so the HNSW search path can
// use it as a ground-truth rerank baseline.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

#include <cmath>

using Catch::Matchers::WithinAbs;
using duckdb::case_insensitive_map_t;
using duckdb::data_t;
using duckdb::idx_t;
using duckdb::Value;
using duckdb::vector;
using duckdb::vindex::CreateQuantizer;
using duckdb::vindex::MetricKind;
using duckdb::vindex::Quantizer;
using duckdb::vindex::QuantizerKind;

namespace {

std::unique_ptr<Quantizer> MakeFlat(MetricKind metric, idx_t dim) {
	case_insensitive_map_t<Value> opts; // no options → defaults to flat
	return CreateQuantizer(opts, metric, dim);
}

} // namespace

TEST_CASE("FlatQuantizer reports flat kind and matching code size", "[quantizer][flat][unit]") {
	auto q = MakeFlat(MetricKind::L2SQ, 8);
	REQUIRE(q->Kind() == QuantizerKind::FLAT);
	REQUIRE(q->Metric() == MetricKind::L2SQ);
	REQUIRE(q->CodeSize() == 8 * sizeof(float));
	REQUIRE(q->QueryWorkspaceSize() == 8);
}

TEST_CASE("FlatQuantizer encode is a verbatim float32 copy", "[quantizer][flat][unit]") {
	auto q = MakeFlat(MetricKind::L2SQ, 4);
	const float vec[] = {1.0f, -2.0f, 3.5f, 0.0f};
	vector<data_t> code(q->CodeSize());
	q->Encode(vec, code.data());
	auto decoded = reinterpret_cast<const float *>(code.data());
	for (idx_t i = 0; i < 4; i++) {
		REQUIRE(decoded[i] == vec[i]);
	}
}

TEST_CASE("FlatQuantizer L2SQ distance matches the naive formula", "[quantizer][flat][unit]") {
	auto q = MakeFlat(MetricKind::L2SQ, 3);
	const float stored[] = {1.0f, 2.0f, 3.0f};
	const float query[] = {4.0f, 6.0f, 8.0f};

	vector<data_t> code(q->CodeSize());
	q->Encode(stored, code.data());

	vector<float> workspace(q->QueryWorkspaceSize());
	q->PreprocessQuery(query, workspace.data());

	// (1-4)^2 + (2-6)^2 + (3-8)^2 = 9 + 16 + 25 = 50
	REQUIRE_THAT(q->EstimateDistance(code.data(), workspace.data()), WithinAbs(50.0f, 1e-5));
}

TEST_CASE("FlatQuantizer IP distance returns negative inner product", "[quantizer][flat][unit]") {
	auto q = MakeFlat(MetricKind::IP, 3);
	const float stored[] = {1.0f, 2.0f, 3.0f};
	const float query[] = {1.0f, 0.0f, -1.0f};

	vector<data_t> code(q->CodeSize());
	q->Encode(stored, code.data());
	vector<float> workspace(q->QueryWorkspaceSize());
	q->PreprocessQuery(query, workspace.data());

	// -(1*1 + 2*0 + 3*-1) = -(-2) = 2
	REQUIRE_THAT(q->EstimateDistance(code.data(), workspace.data()), WithinAbs(2.0f, 1e-5));
}

TEST_CASE("FlatQuantizer cosine distance is in [0, 2] and symmetric", "[quantizer][flat][unit]") {
	auto q = MakeFlat(MetricKind::COSINE, 3);
	const float a[] = {1.0f, 0.0f, 0.0f};
	const float b[] = {1.0f, 0.0f, 0.0f};
	const float orth[] = {0.0f, 1.0f, 0.0f};
	const float anti[] = {-1.0f, 0.0f, 0.0f};

	vector<data_t> code(q->CodeSize());
	vector<float> workspace(q->QueryWorkspaceSize());

	q->Encode(a, code.data());

	q->PreprocessQuery(b, workspace.data());
	REQUIRE_THAT(q->EstimateDistance(code.data(), workspace.data()), WithinAbs(0.0f, 1e-5));

	q->PreprocessQuery(orth, workspace.data());
	REQUIRE_THAT(q->EstimateDistance(code.data(), workspace.data()), WithinAbs(1.0f, 1e-5));

	q->PreprocessQuery(anti, workspace.data());
	REQUIRE_THAT(q->EstimateDistance(code.data(), workspace.data()), WithinAbs(2.0f, 1e-5));
}

TEST_CASE("FlatQuantizer cosine handles zero-norm query without NaN", "[quantizer][flat][unit]") {
	auto q = MakeFlat(MetricKind::COSINE, 3);
	const float stored[] = {1.0f, 2.0f, 3.0f};
	const float zero[] = {0.0f, 0.0f, 0.0f};

	vector<data_t> code(q->CodeSize());
	q->Encode(stored, code.data());
	vector<float> workspace(q->QueryWorkspaceSize());
	q->PreprocessQuery(zero, workspace.data());

	// Dot product is 0 with a zero query, so distance falls back to 1.0.
	REQUIRE_THAT(q->EstimateDistance(code.data(), workspace.data()), WithinAbs(1.0f, 1e-5));
}

TEST_CASE("FlatQuantizer Serialize/Deserialize preserves metric and dim",
          "[quantizer][flat][unit]") {
	auto src = MakeFlat(MetricKind::COSINE, 16);
	vector<data_t> blob;
	src->Serialize(blob);
	REQUIRE(blob.size() > 0);

	auto dst = MakeFlat(MetricKind::L2SQ, 1); // different init values
	dst->Deserialize(blob.data(), blob.size());
	REQUIRE(dst->Metric() == MetricKind::COSINE);
	REQUIRE(dst->CodeSize() == 16 * sizeof(float));
	REQUIRE(dst->QueryWorkspaceSize() == 16);
}

TEST_CASE("CreateQuantizer parses explicit quantizer='flat'", "[quantizer][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("flat");
	auto q = CreateQuantizer(opts, MetricKind::L2SQ, 4);
	REQUIRE(q->Kind() == QuantizerKind::FLAT);
}

TEST_CASE("CreateQuantizer rejects unknown quantizer names", "[quantizer][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("scalar");
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 4), duckdb::BinderException);
}

TEST_CASE("CreateQuantizer rejects non-string quantizer option", "[quantizer][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value::INTEGER(3);
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 4), duckdb::BinderException);
}

// RaBitQ availability + bits-option tests live in test_rabitq_quantizer.cpp.
