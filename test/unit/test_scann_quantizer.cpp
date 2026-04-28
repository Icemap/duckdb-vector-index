// Unit tests for the ScaNN (anisotropic product quantization) quantizer (M4).
//
// Coverage:
//   - Code size / workspace size contracts for bits ∈ {4, 8}.
//   - Constructor rejects cosine, non-divisible m, unsupported bits, eta ≤ 0.
//   - Encode before Train throws.
//   - Recall@10 floor on 1k × 128 Gaussian (bits=8, m=16, 10× rerank) ≥ 0.90.
//   - Serialize/Deserialize round-trip reproduces identical distance estimates
//     (including eta).
//   - PreprocessQuery is deterministic.
//   - CodeDistance(code, code) == 0 for self-pairs.
//   - Factory wiring through CreateQuantizer.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/scann_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

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
using duckdb::vindex::scann::ScannQuantizer;

namespace {

float TrueL2Sq(const float *a, const float *b, idx_t dim) {
	float acc = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		const float d = a[i] - b[i];
		acc += d * d;
	}
	return acc;
}

vector<float> RandomVectors(idx_t n, idx_t dim, uint64_t seed) {
	vector<float> out(n * dim);
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	for (idx_t i = 0; i < n * dim; i++) {
		out[i] = nd(rng);
	}
	return out;
}

} // namespace

TEST_CASE("ScannQuantizer code sizes match m and bits", "[scann][encode][unit]") {
	ScannQuantizer q8(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/8);
	REQUIRE(q8.CodeSize() == 16);
	REQUIRE(q8.QueryWorkspaceSize() == 16 * 256);
	REQUIRE(q8.Kind() == QuantizerKind::SCANN);
	REQUIRE(q8.Metric() == MetricKind::L2SQ);
	REQUIRE(q8.M() == 16);
	REQUIRE(q8.Bits() == 8);
	REQUIRE_THAT(q8.Eta(), WithinAbs(4.0f, 1e-6));

	ScannQuantizer q4(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/4);
	REQUIRE(q4.CodeSize() == 8);
	REQUIRE(q4.QueryWorkspaceSize() == 16 * 16);
}

TEST_CASE("ScannQuantizer rejects bad construction", "[scann][encode][unit]") {
	REQUIRE_THROWS_AS(ScannQuantizer(MetricKind::L2SQ, 130, /*m=*/16, /*bits=*/8),
	                  duckdb::InvalidInputException);
	REQUIRE_THROWS_AS(ScannQuantizer(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/2),
	                  duckdb::InvalidInputException);
	REQUIRE_THROWS_AS(ScannQuantizer(MetricKind::COSINE, 128, /*m=*/16, /*bits=*/8),
	                  duckdb::InvalidInputException);
	REQUIRE_THROWS_AS(ScannQuantizer(MetricKind::L2SQ, 128, /*m=*/0, /*bits=*/8),
	                  duckdb::InvalidInputException);
	REQUIRE_THROWS_AS(ScannQuantizer(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/8, /*eta=*/0.0f),
	                  duckdb::InvalidInputException);
	REQUIRE_THROWS_AS(ScannQuantizer(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/8, /*eta=*/-1.0f),
	                  duckdb::InvalidInputException);
}

TEST_CASE("ScannQuantizer Encode before Train throws", "[scann][encode][unit]") {
	ScannQuantizer q(MetricKind::L2SQ, 8, /*m=*/2, /*bits=*/8);
	float v[8] = {0};
	vector<data_t> code(q.CodeSize());
	REQUIRE_THROWS_AS(q.Encode(v, code.data()), duckdb::InternalException);
}

TEST_CASE("ScannQuantizer bits=8 + 10× rerank achieves Recall@10 ≥ 0.90 on 1k × 128 Gaussian",
          "[scann][recall][unit]") {
	const idx_t n = 1000;
	const idx_t dim = 128;
	const idx_t m = 16;
	const idx_t k = 10;
	const idx_t rerank_mult = 10;
	const idx_t num_queries = 25;

	auto data = RandomVectors(n, dim, /*seed=*/17);
	auto queries = RandomVectors(num_queries, dim, /*seed=*/918273);

	ScannQuantizer q(MetricKind::L2SQ, dim, static_cast<uint8_t>(m), /*bits=*/8);
	q.Train(data.data(), n, dim);

	vector<data_t> codes(n * q.CodeSize());
	for (idx_t i = 0; i < n; i++) {
		q.Encode(data.data() + i * dim, codes.data() + i * q.CodeSize());
	}

	vector<float> workspace(q.QueryWorkspaceSize());
	idx_t total_hits = 0;
	for (idx_t qi = 0; qi < num_queries; qi++) {
		const float *query = queries.data() + qi * dim;
		q.PreprocessQuery(query, workspace.data());

		vector<std::pair<float, idx_t>> est(n);
		for (idx_t i = 0; i < n; i++) {
			est[i] = {q.EstimateDistance(codes.data() + i * q.CodeSize(), workspace.data()), i};
		}
		const idx_t cand_n = std::min<idx_t>(k * rerank_mult, n);
		std::partial_sort(est.begin(), est.begin() + cand_n, est.end(),
		                  [](const auto &a, const auto &b) { return a.first < b.first; });

		vector<std::pair<float, idx_t>> rerank(cand_n);
		for (idx_t i = 0; i < cand_n; i++) {
			const idx_t row = est[i].second;
			rerank[i] = {TrueL2Sq(query, data.data() + row * dim, dim), row};
		}
		std::partial_sort(rerank.begin(), rerank.begin() + k, rerank.end(),
		                  [](const auto &a, const auto &b) { return a.first < b.first; });

		vector<std::pair<float, idx_t>> truth(n);
		for (idx_t i = 0; i < n; i++) {
			truth[i] = {TrueL2Sq(query, data.data() + i * dim, dim), i};
		}
		std::partial_sort(truth.begin(), truth.begin() + k, truth.end(),
		                  [](const auto &a, const auto &b) { return a.first < b.first; });

		for (idx_t a = 0; a < k; a++) {
			for (idx_t b = 0; b < k; b++) {
				if (rerank[a].second == truth[b].second) {
					total_hits++;
					break;
				}
			}
		}
	}
	const double recall = static_cast<double>(total_hits) / (num_queries * k);
	INFO("Recall@10 (ScaNN m=16 bits=8 eta=4, 10× rerank) = " << recall);
	REQUIRE(recall >= 0.90);
}

TEST_CASE("ScannQuantizer PreprocessQuery is deterministic", "[scann][query][unit]") {
	const idx_t dim = 32;
	auto data = RandomVectors(256, dim, 7);
	ScannQuantizer q(MetricKind::L2SQ, dim, /*m=*/8, /*bits=*/8);
	q.Train(data.data(), 256, dim);

	vector<float> ws_a(q.QueryWorkspaceSize());
	vector<float> ws_b(q.QueryWorkspaceSize());
	q.PreprocessQuery(data.data(), ws_a.data());
	q.PreprocessQuery(data.data(), ws_b.data());
	REQUIRE(std::memcmp(ws_a.data(), ws_b.data(), ws_a.size() * sizeof(float)) == 0);
}

TEST_CASE("ScannQuantizer Serialize round-trip reproduces estimates", "[scann][persistence][unit]") {
	const idx_t dim = 32;
	auto data = RandomVectors(256, dim, 11);

	for (uint8_t bits : {4, 8}) {
		ScannQuantizer src(MetricKind::L2SQ, dim, /*m=*/8, bits, /*eta=*/2.5f);
		src.Train(data.data(), 256, dim);

		vector<data_t> code(src.CodeSize());
		src.Encode(data.data(), code.data());

		vector<float> workspace_src(src.QueryWorkspaceSize());
		src.PreprocessQuery(data.data() + dim, workspace_src.data());
		const float est_src = src.EstimateDistance(code.data(), workspace_src.data());

		vector<data_t> blob;
		src.Serialize(blob);

		ScannQuantizer dst(MetricKind::L2SQ, 8, /*m=*/2, /*bits=*/4);
		dst.Deserialize(blob.data(), blob.size());
		REQUIRE(dst.Bits() == bits);
		REQUIRE(dst.M() == 8);
		REQUIRE(dst.Metric() == MetricKind::L2SQ);
		REQUIRE_THAT(dst.Eta(), WithinAbs(2.5f, 1e-6));

		vector<float> workspace_dst(dst.QueryWorkspaceSize());
		dst.PreprocessQuery(data.data() + dim, workspace_dst.data());
		const float est_dst = dst.EstimateDistance(code.data(), workspace_dst.data());
		REQUIRE_THAT(est_dst, WithinAbs(est_src, 1e-5));
	}
}

TEST_CASE("ScannQuantizer CodeDistance is zero for self-pairs", "[scann][code-distance][unit]") {
	const idx_t dim = 32;
	auto data = RandomVectors(128, dim, 5);
	ScannQuantizer q(MetricKind::L2SQ, dim, /*m=*/8, /*bits=*/8);
	q.Train(data.data(), 128, dim);

	vector<data_t> code_a(q.CodeSize());
	vector<data_t> code_b(q.CodeSize());
	q.Encode(data.data(), code_a.data());
	q.Encode(data.data(), code_b.data());
	REQUIRE_THAT(q.CodeDistance(code_a.data(), code_b.data()), WithinAbs(0.0f, 1e-5));

	q.Encode(data.data() + dim * 5, code_b.data());
	REQUIRE(q.CodeDistance(code_a.data(), code_b.data()) > 0.0f);
}

TEST_CASE("CreateQuantizer wires scann with defaults", "[scann][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("scann");
	auto q = CreateQuantizer(opts, MetricKind::L2SQ, 128);
	REQUIRE(q->Kind() == QuantizerKind::SCANN);
	REQUIRE(q->CodeSize() == 32); // m=dim/4=32, bits=8
}

TEST_CASE("CreateQuantizer scann accepts explicit m, bits, eta", "[scann][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("scann");
	opts["m"] = Value::INTEGER(16);
	opts["bits"] = Value::INTEGER(4);
	opts["eta"] = Value::DOUBLE(2.0);
	auto q = CreateQuantizer(opts, MetricKind::L2SQ, 128);
	REQUIRE(q->Kind() == QuantizerKind::SCANN);
	REQUIRE(q->CodeSize() == 8);
}

TEST_CASE("CreateQuantizer scann rejects bad options", "[scann][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("scann");
	opts["bits"] = Value::INTEGER(3);
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 128), duckdb::BinderException);

	opts["bits"] = Value::INTEGER(8);
	opts["m"] = Value::INTEGER(17);
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 128), duckdb::BinderException);

	case_insensitive_map_t<Value> eta_opts;
	eta_opts["quantizer"] = Value("scann");
	eta_opts["eta"] = Value::DOUBLE(0.0);
	REQUIRE_THROWS_AS(CreateQuantizer(eta_opts, MetricKind::L2SQ, 128), duckdb::BinderException);

	case_insensitive_map_t<Value> cos_opts;
	cos_opts["quantizer"] = Value("scann");
	REQUIRE_THROWS_AS(CreateQuantizer(cos_opts, MetricKind::COSINE, 128), duckdb::BinderException);
}
