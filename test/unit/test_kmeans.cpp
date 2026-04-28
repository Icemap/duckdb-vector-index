// Unit tests for src/algo/ivf/kmeans.cpp — k-means++ + Lloyd.
//
// The algorithm is stochastic (seeded), so we test invariants rather than
// specific centroid coordinates:
//
//   1. Output shape: exactly k × dim floats populated.
//   2. Convergence: on a dataset generated from k well-separated clusters,
//      every produced centroid lies near one of the true cluster means.
//   3. Determinism: same seed → identical centroids byte-for-byte.

#include <catch2/catch_test_macros.hpp>

#include "algo/ivf/kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using duckdb::idx_t;
using duckdb::vindex::ivf::KMeansPlusPlus;

namespace {

std::vector<float> MakeClustered(idx_t k, idx_t per_cluster, idx_t dim, uint64_t seed,
                                 std::vector<std::vector<float>> &out_means) {
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> noise(0.0f, 0.1f);
	std::uniform_real_distribution<float> center_coord(-10.0f, 10.0f);

	out_means.assign(k, std::vector<float>(dim));
	for (idx_t c = 0; c < k; c++) {
		for (idx_t d = 0; d < dim; d++) {
			out_means[c][d] = center_coord(rng);
		}
	}

	std::vector<float> data(k * per_cluster * dim);
	for (idx_t c = 0; c < k; c++) {
		for (idx_t i = 0; i < per_cluster; i++) {
			for (idx_t d = 0; d < dim; d++) {
				data[(c * per_cluster + i) * dim + d] = out_means[c][d] + noise(rng);
			}
		}
	}
	return data;
}

float L2sq(const float *a, const float *b, idx_t dim) {
	float acc = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		const float d = a[i] - b[i];
		acc += d * d;
	}
	return acc;
}

} // namespace

TEST_CASE("KMeans converges on well-separated clusters", "[kmeans][unit]") {
	const idx_t k = 8;
	const idx_t per_cluster = 64;
	const idx_t dim = 16;

	std::vector<std::vector<float>> true_means;
	auto data = MakeClustered(k, per_cluster, dim, /*seed=*/42, true_means);

	std::vector<float> centroids(k * dim, 0.0f);
	KMeansPlusPlus(data.data(), k * per_cluster, dim, k, /*seed=*/7, /*iters=*/20, centroids.data());

	// Every produced centroid should map to exactly one true mean within the
	// cluster radius (noise σ=0.1 → squared radius ≈ dim × 0.01 ≈ 0.16).
	std::vector<bool> matched(k, false);
	for (idx_t c = 0; c < k; c++) {
		float best = std::numeric_limits<float>::infinity();
		idx_t best_m = 0;
		for (idx_t m = 0; m < k; m++) {
			const float d = L2sq(centroids.data() + c * dim, true_means[m].data(), dim);
			if (d < best) {
				best = d;
				best_m = m;
			}
		}
		REQUIRE(best < 1.0f);
		matched[best_m] = true;
	}
	for (idx_t m = 0; m < k; m++) {
		REQUIRE(matched[m]);
	}
}

TEST_CASE("KMeans is deterministic for a fixed seed", "[kmeans][unit]") {
	const idx_t n = 512;
	const idx_t dim = 8;
	const idx_t k = 16;

	std::mt19937_64 rng(123);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	std::vector<float> data(n * dim);
	for (auto &v : data) {
		v = nd(rng);
	}

	std::vector<float> a(k * dim), b(k * dim);
	KMeansPlusPlus(data.data(), n, dim, k, /*seed=*/99, 20, a.data());
	KMeansPlusPlus(data.data(), n, dim, k, /*seed=*/99, 20, b.data());
	REQUIRE(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("KMeans handles n < k by repeating samples", "[kmeans][unit]") {
	const idx_t dim = 4;
	const idx_t n = 3;
	const idx_t k = 5;
	std::vector<float> data = {
	    1.0f, 0.0f, 0.0f, 0.0f, //
	    0.0f, 1.0f, 0.0f, 0.0f, //
	    0.0f, 0.0f, 1.0f, 0.0f, //
	};

	std::vector<float> centroids(k * dim, -1.0f);
	REQUIRE_NOTHROW(KMeansPlusPlus(data.data(), n, dim, k, /*seed=*/0, 5, centroids.data()));
	// Every produced centroid must match one of the input rows.
	for (idx_t c = 0; c < k; c++) {
		bool ok = false;
		for (idx_t i = 0; i < n; i++) {
			if (L2sq(centroids.data() + c * dim, data.data() + i * dim, dim) < 1e-6f) {
				ok = true;
				break;
			}
		}
		REQUIRE(ok);
	}
}
