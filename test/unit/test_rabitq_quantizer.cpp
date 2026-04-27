// Unit tests for the b-bit scalar RaBitQ quantizer (task #17 + #18).
//
// The tests here cover:
//   - Encode/Estimate API contract (code size, trained-before-use guard).
//   - Distance ordering: ranking N candidates by EstimateDistance should mostly
//     agree with ranking by true L2SQ, on a well-separated synthetic fixture.
//   - Recall@10 floors on 1k×128 Gaussian:
//       * bits=1 + 20× rerank ≥ 0.90
//       * bits=3 + 10× rerank ≥ 0.99 (AGENTS.md §6.3 synthetic floor; #22 moves
//         the SIFT1M regression target, also ≥ 0.99).
//   - Serialize/Deserialize round-trip reproduces identical distance estimates
//     at every supported bit width.
//   - Factory wiring through CreateQuantizer, default bits=3, accepts
//     {1,2,3,4,5,7,8}, rejects 0/6/9/...

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/vector.hpp"

#include "../../src/quant/rabitq/rabitq_quantizer.hpp"
#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

#include <algorithm>
#include <cmath>
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
using duckdb::vindex::rabitq::RabitqQuantizer;

namespace {

float TrueL2Sq(const float *a, const float *b, idx_t dim) {
	float acc = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		const float d = a[i] - b[i];
		acc += d * d;
	}
	return acc;
}

// Generate n vectors of dim floats from a fixed-seed Gaussian. Returns a flat
// row-major buffer.
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

TEST_CASE("RabitqQuantizer 1-bit code size matches packed-bits + trailer",
          "[rabitq][encode][unit]") {
	RabitqQuantizer q(MetricKind::L2SQ, 16, /*bits=*/1);
	// 16 / 8 = 2 bytes of bits + 8 bytes trailer (alpha, scale).
	REQUIRE(q.CodeSize() == 2 + 8);
	REQUIRE(q.QueryWorkspaceSize() == 16 + 1);
	REQUIRE(q.Kind() == QuantizerKind::RABITQ);
	REQUIRE(q.Metric() == MetricKind::L2SQ);
	REQUIRE(q.Bits() == 1);
}

TEST_CASE("RabitqQuantizer b-bit code size scales with bits", "[rabitq][encode][unit]") {
	// dim=32, bits=3: ceil(32*3/8) = 12 bytes + 8 trailer = 20 B.
	RabitqQuantizer q3(MetricKind::L2SQ, 32, /*bits=*/3);
	REQUIRE(q3.CodeSize() == 12 + 8);

	// dim=128, bits=8: 128 bytes + 8 trailer.
	RabitqQuantizer q8(MetricKind::L2SQ, 128, /*bits=*/8);
	REQUIRE(q8.CodeSize() == 128 + 8);

	// dim=130, bits=1: ceil(130/8) = 17 bytes.
	RabitqQuantizer q1(MetricKind::L2SQ, 130, /*bits=*/1);
	REQUIRE(q1.CodeSize() == 17 + 8);

	// dim=130, bits=5: ceil(130*5/8) = ceil(650/8) = 82 bytes.
	RabitqQuantizer q5(MetricKind::L2SQ, 130, /*bits=*/5);
	REQUIRE(q5.CodeSize() == 82 + 8);
}

TEST_CASE("RabitqQuantizer rejects unsupported bit widths", "[rabitq][encode][unit]") {
	REQUIRE_THROWS_AS(RabitqQuantizer(MetricKind::L2SQ, 8, /*bits=*/0), duckdb::NotImplementedException);
	REQUIRE_THROWS_AS(RabitqQuantizer(MetricKind::L2SQ, 8, /*bits=*/6), duckdb::NotImplementedException);
	REQUIRE_THROWS_AS(RabitqQuantizer(MetricKind::L2SQ, 8, /*bits=*/9), duckdb::NotImplementedException);
	REQUIRE_THROWS_AS(RabitqQuantizer(MetricKind::L2SQ, 8, /*bits=*/16), duckdb::NotImplementedException);
}

TEST_CASE("RabitqQuantizer rejects IP/cosine until task #19", "[rabitq][encode][unit]") {
	REQUIRE_THROWS_AS(RabitqQuantizer(MetricKind::IP, 8), duckdb::NotImplementedException);
	REQUIRE_THROWS_AS(RabitqQuantizer(MetricKind::COSINE, 8), duckdb::NotImplementedException);
}

TEST_CASE("RabitqQuantizer Encode before Train throws", "[rabitq][encode][unit]") {
	RabitqQuantizer q(MetricKind::L2SQ, 8);
	float v[8] = {0};
	vector<data_t> code(q.CodeSize());
	REQUIRE_THROWS_AS(q.Encode(v, code.data()), duckdb::InternalException);
}

TEST_CASE("RabitqQuantizer trains centroid from sample mean", "[rabitq][train][unit]") {
	RabitqQuantizer q(MetricKind::L2SQ, 4);
	const float samples[] = {
	    1.0f, 2.0f, 3.0f, 4.0f, //
	    3.0f, 4.0f, 5.0f, 6.0f, //
	    5.0f, 6.0f, 7.0f, 8.0f  //
	};
	q.Train(samples, 3, 4);
	REQUIRE(q.IsTrained());
	const auto &c = q.Centroid();
	REQUIRE(c.size() == 4);
	REQUIRE_THAT(c[0], WithinAbs(3.0f, 1e-5));
	REQUIRE_THAT(c[1], WithinAbs(4.0f, 1e-5));
	REQUIRE_THAT(c[2], WithinAbs(5.0f, 1e-5));
	REQUIRE_THAT(c[3], WithinAbs(6.0f, 1e-5));
}

namespace {

// Returns Recall@k of the codec at the given rerank multiplier over a
// 1k×128 Gaussian fixture. Factored out so bits=1 and bits=3 share one harness.
double MeasureRerankRecall(uint8_t bits, idx_t rerank_mult, uint64_t data_seed = 17,
                           uint64_t query_seed = 918273) {
	const idx_t n = 1000;
	const idx_t dim = 128;
	const idx_t k = 10;
	const idx_t num_queries = 25;

	auto data = RandomVectors(n, dim, data_seed);
	auto queries = RandomVectors(num_queries, dim, query_seed);

	RabitqQuantizer q(MetricKind::L2SQ, dim, bits);
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

		// 1) Rank all rows by *estimated* distance, take top k × rerank_mult.
		vector<std::pair<float, idx_t>> est(n);
		for (idx_t i = 0; i < n; i++) {
			est[i] = {q.EstimateDistance(codes.data() + i * q.CodeSize(), workspace.data()), i};
		}
		const idx_t cand_n = std::min<idx_t>(k * rerank_mult, n);
		std::partial_sort(est.begin(), est.begin() + cand_n, est.end(),
		                  [](const auto &a, const auto &b) { return a.first < b.first; });

		// 2) Re-rank those candidates with true L2SQ → final top k.
		vector<std::pair<float, idx_t>> rerank(cand_n);
		for (idx_t i = 0; i < cand_n; i++) {
			const idx_t row = est[i].second;
			rerank[i] = {TrueL2Sq(query, data.data() + row * dim, dim), row};
		}
		std::partial_sort(rerank.begin(), rerank.begin() + k, rerank.end(),
		                  [](const auto &a, const auto &b) { return a.first < b.first; });

		// Ground truth: top k by true L2SQ over the full set.
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
	return static_cast<double>(total_hits) / (num_queries * k);
}

} // namespace

// 1-bit RaBitQ has ~40% Recall@10 on its own; the production path (see
// doc/architecture.md §3) takes the top `k × rerank_multiple` candidates by
// estimated distance and re-ranks them by exact L2 using the original ARRAY
// column. With rerank_multiple=20 the 1-bit estimator should recover ≥90% of
// the true top-10 — this matches AGENTS.md §6.3 "RaBitQ 1-bit + HNSW" floor.
TEST_CASE("RabitqQuantizer bits=1 + 20× rerank achieves Recall@10 ≥ 0.90 on 1k×128 Gaussian",
          "[rabitq][recall][unit]") {
	const double recall = MeasureRerankRecall(/*bits=*/1, /*rerank=*/20);
	INFO("Recall@10 (1-bit + 20× rerank) = " << recall);
	REQUIRE(recall >= 0.90);
}

// bits=3 is the production default. AGENTS.md §6.3 sets the SIFT1M floor at
// ≥0.99 with 10× rerank; this synthetic fixture is the pre-SIFT smoke test.
TEST_CASE("RabitqQuantizer bits=3 + 10× rerank achieves Recall@10 ≥ 0.99 on 1k×128 Gaussian",
          "[rabitq][recall][unit]") {
	const double recall = MeasureRerankRecall(/*bits=*/3, /*rerank=*/10);
	INFO("Recall@10 (3-bit + 10× rerank) = " << recall);
	REQUIRE(recall >= 0.99);
}

// Sanity check that the estimator monotonically improves with more bits, even
// without rerank. This catches packing / unpacking / scale bugs where a higher
// bit-width regresses below the 1-bit baseline.
TEST_CASE("RabitqQuantizer recall (no rerank) improves with bits", "[rabitq][recall][unit]") {
	const double r1 = MeasureRerankRecall(/*bits=*/1, /*rerank=*/1);
	const double r3 = MeasureRerankRecall(/*bits=*/3, /*rerank=*/1);
	const double r8 = MeasureRerankRecall(/*bits=*/8, /*rerank=*/1);
	INFO("Recall@10 without rerank: bits=1 " << r1 << ", bits=3 " << r3 << ", bits=8 " << r8);
	REQUIRE(r3 >= r1);
	REQUIRE(r8 >= r3);
	// bits=8 without rerank should be very close to exact.
	REQUIRE(r8 >= 0.95);
}

TEST_CASE("RabitqQuantizer Serialize round-trip reproduces distance estimates at every bit width",
          "[rabitq][persistence][unit]") {
	const idx_t dim = 32;
	auto data = RandomVectors(64, dim, 7);

	for (uint8_t bits : {1, 2, 3, 4, 5, 7, 8}) {
		RabitqQuantizer src(MetricKind::L2SQ, dim, bits);
		src.Train(data.data(), 64, dim);

		vector<data_t> code(src.CodeSize());
		src.Encode(data.data(), code.data()); // encode row 0
		vector<float> workspace_src(src.QueryWorkspaceSize());
		src.PreprocessQuery(data.data() + dim, workspace_src.data()); // query = row 1
		const float est_src = src.EstimateDistance(code.data(), workspace_src.data());

		vector<data_t> blob;
		src.Serialize(blob);

		// Deserialize overwrites bits_/dim_/metric_ from the blob, so any init
		// values work — use a different bit width to prove the disk value wins.
		RabitqQuantizer dst(MetricKind::L2SQ, 1, bits == 1 ? 8 : 1);
		dst.Deserialize(blob.data(), blob.size());
		REQUIRE(dst.IsTrained());
		REQUIRE(dst.Bits() == bits);
		REQUIRE(dst.CodeSize() == src.CodeSize());
		REQUIRE(dst.QueryWorkspaceSize() == src.QueryWorkspaceSize());

		// A freshly encoded code from `dst` should match the one `src` produced —
		// same centroid, same rotation matrix, same bits.
		vector<data_t> code_dst(dst.CodeSize());
		dst.Encode(data.data(), code_dst.data());
		REQUIRE(std::memcmp(code.data(), code_dst.data(), code.size()) == 0);

		vector<float> workspace_dst(dst.QueryWorkspaceSize());
		dst.PreprocessQuery(data.data() + dim, workspace_dst.data());
		const float est_dst = dst.EstimateDistance(code.data(), workspace_dst.data());
		REQUIRE_THAT(est_dst, WithinAbs(est_src, 1e-4f));
	}
}

TEST_CASE("CreateQuantizer('rabitq') defaults to bits=3 (README §'Quantizer bits vs recall')",
          "[rabitq][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("rabitq");
	auto q = CreateQuantizer(opts, MetricKind::L2SQ, 8);
	REQUIRE(q->Kind() == QuantizerKind::RABITQ);
	// dim=8, bits=3: ceil(8*3/8) = 3 bytes + 8 trailer.
	REQUIRE(q->CodeSize() == 3 + 8);
}

TEST_CASE("CreateQuantizer('rabitq') accepts every supported bit width",
          "[rabitq][factory][unit]") {
	for (int b : {1, 2, 3, 4, 5, 7, 8}) {
		case_insensitive_map_t<Value> opts;
		opts["quantizer"] = Value("rabitq");
		opts["bits"] = Value::INTEGER(b);
		auto q = CreateQuantizer(opts, MetricKind::L2SQ, 16);
		INFO("bits=" << b);
		REQUIRE(q->Kind() == QuantizerKind::RABITQ);
	}
}

TEST_CASE("CreateQuantizer('rabitq') rejects unsupported bit widths",
          "[rabitq][factory][unit]") {
	for (int b : {0, 6, 9, 10, 16, -1}) {
		case_insensitive_map_t<Value> opts;
		opts["quantizer"] = Value("rabitq");
		opts["bits"] = Value::INTEGER(b);
		INFO("bits=" << b);
		REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 8), duckdb::BinderException);
	}
}

TEST_CASE("CreateQuantizer rejects non-integer bits option",
          "[rabitq][factory][unit]") {
	case_insensitive_map_t<Value> opts;
	opts["quantizer"] = Value("rabitq");
	opts["bits"] = Value("three");
	REQUIRE_THROWS_AS(CreateQuantizer(opts, MetricKind::L2SQ, 8), duckdb::BinderException);
}
