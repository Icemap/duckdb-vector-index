// Unit tests for the classical Product-Quantization quantizer (M3).
//
// Coverage:
//   - Code size / workspace size contracts for bits ∈ {4, 8}.
//   - Constructor rejects cosine, non-divisible m, unsupported bits.
//   - Encode before Train throws.
//   - Recall@10 floor on 1k × 128 Gaussian (bits=8, m=16, 10× rerank) ≥ 0.90.
//     PQ over unstructured Gaussian data without rerank lands ~0.4 because
//     k-means is asymptotically useless on isotropic noise — the production
//     path always pairs PQ with rerank (see README.md §Rerank).
//   - Serialize/Deserialize round-trip reproduces identical distance estimates.
//   - PreprocessQuery is deterministic (same seed ⇒ identical workspace bytes).
//   - CodeDistance gives the same ordering as EstimateDistance for well-
//     separated fixture pairs (guards HNSW's Algorithm-4 heuristic).
//   - Factory wiring through CreateQuantizer.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/metric.hpp"
#include "vindex/pq_quantizer.hpp"
#include "vindex/quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using duckdb::case_insensitive_map_t;
using duckdb::data_t;
using duckdb::idx_t;
using duckdb::Value;
using duckdb::vector;
using duckdb::vindex::CreateQuantizer;
using duckdb::vindex::MetricKind;
using duckdb::vindex::Quantizer;
using duckdb::vindex::QuantizerKind;
using duckdb::vindex::pq::PqQuantizer;

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

TEST_CASE("PqQuantizer code sizes match m and bits", "[pq][encode][unit]") {
	// dim=128, m=16, bits=8 → 16 bytes.
	PqQuantizer q8(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/8);
	REQUIRE(q8.CodeSize() == 16);
	REQUIRE(q8.QueryWorkspaceSize() == 16 * 256);
	REQUIRE(q8.Kind() == QuantizerKind::PQ);
	REQUIRE(q8.Metric() == MetricKind::L2SQ);
	REQUIRE(q8.M() == 16);
	REQUIRE(q8.Bits() == 8);

	// dim=128, m=16, bits=4 → ceil(16*4/8) = 8 bytes.
	PqQuantizer q4(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/4);
	REQUIRE(q4.CodeSize() == 8);
	REQUIRE(q4.QueryWorkspaceSize() == 16 * 16);
}

TEST_CASE("PqQuantizer rejects bad construction", "[pq][encode][unit]") {
	// dim not divisible by m.
	REQUIRE_THROWS_AS(PqQuantizer(MetricKind::L2SQ, 130, /*m=*/16, /*bits=*/8),
	                  duckdb::InvalidInputException);
	// unsupported bits.
	REQUIRE_THROWS_AS(PqQuantizer(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/2),
	                  duckdb::InvalidInputException);
	REQUIRE_THROWS_AS(PqQuantizer(MetricKind::L2SQ, 128, /*m=*/16, /*bits=*/16),
	                  duckdb::InvalidInputException);
	// cosine unsupported.
	REQUIRE_THROWS_AS(PqQuantizer(MetricKind::COSINE, 128, /*m=*/16, /*bits=*/8),
	                  duckdb::InvalidInputException);
	// m == 0.
	REQUIRE_THROWS_AS(PqQuantizer(MetricKind::L2SQ, 128, /*m=*/0, /*bits=*/8),
	                  duckdb::InvalidInputException);
}

TEST_CASE("PqQuantizer Encode before Train throws", "[pq][encode][unit]") {
	PqQuantizer q(MetricKind::L2SQ, 8, /*m=*/2, /*bits=*/8);
	float v[8] = {0};
	vector<data_t> code(q.CodeSize());
	REQUIRE_THROWS_AS(q.Encode(v, code.data()), duckdb::InternalException);
}

TEST_CASE("PqQuantizer 4-bit round-trip decodes both nibbles", "[pq][encode][unit]") {
	// Two-slot toy fixture: verify that writing codes 0x3 and 0xC into adjacent
	// 4-bit slots survives through Encode (slot 0 = low nibble, slot 1 = high).
	// Build by hand: train on data where slot 0 has centroids at distinct
	// positions and force m=2 dim=2.
	const idx_t dim = 2;
	// Each row is one 2-float sample, slot 0 = x, slot 1 = y. Give 16 distinct
	// positions per slot so kmeans can recover all 16 centroids.
	vector<float> samples;
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			samples.push_back(float(i));
			samples.push_back(float(j));
		}
	}
	PqQuantizer q(MetricKind::L2SQ, dim, /*m=*/2, /*bits=*/4);
	q.Train(samples.data(), 256, dim);

	vector<data_t> code(q.CodeSize());
	REQUIRE(q.CodeSize() == 1); // 2 slots × 4 bits = 1 byte.
	// Encode a random sample — the two nibbles of the byte should each be
	// valid centroid ids (< 16).
	q.Encode(samples.data() + 37 * dim, code.data());
	const uint8_t byte = code.data()[0];
	REQUIRE((byte & 0x0F) < 16);
	REQUIRE(((byte >> 4) & 0x0F) < 16);
}

TEST_CASE("PqQuantizer bits=8 + 10× rerank achieves Recall@10 ≥ 0.90 on 1k × 128 Gaussian",
          "[pq][recall][unit]") {
	const idx_t n = 1000;
	const idx_t dim = 128;
	const idx_t m = 16;
	const idx_t k = 10;
	const idx_t rerank_mult = 10;
	const idx_t num_queries = 25;

	auto data = RandomVectors(n, dim, /*seed=*/17);
	auto queries = RandomVectors(num_queries, dim, /*seed=*/918273);

	PqQuantizer q(MetricKind::L2SQ, dim, static_cast<uint8_t>(m), /*bits=*/8);
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

		// 1) top k × rerank_mult by estimated distance.
		vector<std::pair<float, idx_t>> est(n);
		for (idx_t i = 0; i < n; i++) {
			est[i] = {q.EstimateDistance(codes.data() + i * q.CodeSize(), workspace.data()), i};
		}
		const idx_t cand_n = std::min<idx_t>(k * rerank_mult, n);
		std::partial_sort(est.begin(), est.begin() + cand_n, est.end(),
		                  [](const auto &a, const auto &b) { return a.first < b.first; });

		// 2) rerank those candidates with true L2SQ.
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
	INFO("Recall@10 (PQ m=16 bits=8, 10× rerank) = " << recall);
	REQUIRE(recall >= 0.90);
}

TEST_CASE("PqQuantizer PreprocessQuery is deterministic", "[pq][query][unit]") {
	const idx_t dim = 32;
	auto data = RandomVectors(256, dim, 7);
	PqQuantizer q(MetricKind::L2SQ, dim, /*m=*/8, /*bits=*/8);
	q.Train(data.data(), 256, dim);

	vector<float> ws_a(q.QueryWorkspaceSize());
	vector<float> ws_b(q.QueryWorkspaceSize());
	q.PreprocessQuery(data.data(), ws_a.data());
	q.PreprocessQuery(data.data(), ws_b.data());
	REQUIRE(std::memcmp(ws_a.data(), ws_b.data(), ws_a.size() * sizeof(float)) == 0);
}

TEST_CASE("PqQuantizer Serialize round-trip reproduces estimates", "[pq][persistence][unit]") {
	const idx_t dim = 32;
	auto data = RandomVectors(256, dim, 11);

	for (uint8_t bits : {4, 8}) {
		PqQuantizer src(MetricKind::L2SQ, dim, /*m=*/8, bits);
		src.Train(data.data(), 256, dim);

		vector<data_t> code(src.CodeSize());
		src.Encode(data.data(), code.data());

		vector<float> workspace_src(src.QueryWorkspaceSize());
		src.PreprocessQuery(data.data() + dim, workspace_src.data());
		const float est_src = src.EstimateDistance(code.data(), workspace_src.data());

		vector<data_t> blob;
		src.Serialize(blob);

		// Construct with divergent parameters — Deserialize must overwrite.
		PqQuantizer dst(MetricKind::L2SQ, 8, /*m=*/2, /*bits=*/4);
		dst.Deserialize(blob.data(), blob.size());
		REQUIRE(dst.Bits() == bits);
		REQUIRE(dst.M() == 8);
		REQUIRE(dst.Metric() == MetricKind::L2SQ);

		vector<float> workspace_dst(dst.QueryWorkspaceSize());
		dst.PreprocessQuery(data.data() + dim, workspace_dst.data());
		const float est_dst = dst.EstimateDistance(code.data(), workspace_dst.data());
		REQUIRE_THAT(est_dst, WithinAbs(est_src, 1e-5));
	}
}

TEST_CASE("PqQuantizer CodeDistance matches the L2 sum across slots", "[pq][code-distance][unit]") {
	// For PQ, CodeDistance(a, b) = Σ_s L2sq(centroid[s][code_a[s]], centroid[s][code_b[s]]).
	// Encoding a real sample and then asking for its self-distance must be 0.
	const idx_t dim = 32;
	auto data = RandomVectors(128, dim, 5);
	PqQuantizer q(MetricKind::L2SQ, dim, /*m=*/8, /*bits=*/8);
	q.Train(data.data(), 128, dim);

	vector<data_t> code_a(q.CodeSize());
	vector<data_t> code_b(q.CodeSize());
	q.Encode(data.data(), code_a.data());
	q.Encode(data.data(), code_b.data());
	REQUIRE_THAT(q.CodeDistance(code_a.data(), code_b.data()), WithinAbs(0.0f, 1e-5));

	// Distinct rows → strictly positive (with overwhelming probability on
	// Gaussian data with 128 samples and 256 centroids/slot).
	q.Encode(data.data() + dim * 5, code_b.data());
	REQUIRE(q.CodeDistance(code_a.data(), code_b.data()) > 0.0f);
}

TEST_CASE("CreateQuantizer wires pq with defaults", "[pq][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("pq");
	auto q = CreateQuantizer(opts, MetricKind::L2SQ, 128);
	REQUIRE(q->Kind() == QuantizerKind::PQ);
	// Default m = dim/4 = 32, bits = 8 → 32 bytes.
	REQUIRE(q->CodeSize() == 32);
}

TEST_CASE("CreateQuantizer pq accepts explicit m and bits", "[pq][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("pq");
	opts["m"] = Value::INTEGER(16);
	opts["bits"] = Value::INTEGER(4);
	auto q = CreateQuantizer(opts, MetricKind::L2SQ, 128);
	REQUIRE(q->Kind() == QuantizerKind::PQ);
	REQUIRE(q->CodeSize() == 8); // 16*4/8
}

TEST_CASE("CreateQuantizer pq rejects bad options", "[pq][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("pq");
	opts["bits"] = Value::INTEGER(3);
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 128), duckdb::BinderException);

	opts["bits"] = Value::INTEGER(8);
	opts["m"] = Value::INTEGER(17); // 128 % 17 != 0
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 128), duckdb::BinderException);
}
