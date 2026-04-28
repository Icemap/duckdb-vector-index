#include "algo/ivf/kmeans.hpp"

#include "duckdb/common/exception.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace duckdb {
namespace vindex {
namespace ivf {

namespace {

inline float L2sq(const float *a, const float *b, idx_t dim) {
	simsimd_distance_t d = 0;
	simsimd_l2sq_f32(reinterpret_cast<const simsimd_f32_t *>(a), reinterpret_cast<const simsimd_f32_t *>(b), dim, &d);
	return float(d);
}

// k-means++ seeding: pick the first centroid uniformly at random, then each
// subsequent one with probability proportional to D(x)² (distance to the
// nearest already-chosen centroid).
void KMeansPlusPlusSeed(const float *data, idx_t n, idx_t dim, idx_t k, std::mt19937_64 &rng, float *centroids_out) {
	std::vector<float> min_d2(n, std::numeric_limits<float>::infinity());

	// First centroid: uniform.
	std::uniform_int_distribution<idx_t> first(0, n - 1);
	const idx_t i0 = first(rng);
	std::memcpy(centroids_out, data + i0 * dim, dim * sizeof(float));

	for (idx_t c = 1; c < k; c++) {
		const float *prev = centroids_out + (c - 1) * dim;
		double sum = 0.0;
		for (idx_t i = 0; i < n; i++) {
			const float d = L2sq(data + i * dim, prev, dim);
			if (d < min_d2[i]) {
				min_d2[i] = d;
			}
			sum += min_d2[i];
		}

		if (sum <= 0.0) {
			// All remaining points coincide with an existing centroid —
			// fall back to a random sample to fill up to k.
			std::uniform_int_distribution<idx_t> u(0, n - 1);
			std::memcpy(centroids_out + c * dim, data + u(rng) * dim, dim * sizeof(float));
			continue;
		}

		std::uniform_real_distribution<double> pick(0.0, sum);
		double r = pick(rng);
		idx_t chosen = n - 1;
		for (idx_t i = 0; i < n; i++) {
			r -= min_d2[i];
			if (r <= 0.0) {
				chosen = i;
				break;
			}
		}
		std::memcpy(centroids_out + c * dim, data + chosen * dim, dim * sizeof(float));
	}
}

} // namespace

void KMeansPlusPlus(const float *data, idx_t n, idx_t dim, idx_t k, uint64_t seed, idx_t max_iters,
                    float *centroids_out) {
	if (k == 0 || dim == 0) {
		throw InternalException("KMeansPlusPlus: dim and k must be positive");
	}
	if (n == 0) {
		// Degenerate: leave centroids_out zeroed. Caller's clustering will
		// assign everything to centroid 0.
		std::memset(centroids_out, 0, k * dim * sizeof(float));
		return;
	}

	std::mt19937_64 rng(seed);

	// If n < k, just repeat samples: can't make more unique centroids than
	// points. (Unit tests should ensure n ≥ k.)
	if (n < k) {
		for (idx_t c = 0; c < k; c++) {
			std::memcpy(centroids_out + c * dim, data + (c % n) * dim, dim * sizeof(float));
		}
	} else {
		KMeansPlusPlusSeed(data, n, dim, k, rng, centroids_out);
	}

	std::vector<idx_t> assign(n, 0);
	std::vector<double> sum_vec(k * dim, 0.0);
	std::vector<idx_t> count(k, 0);

	for (idx_t iter = 0; iter < max_iters; iter++) {
		// 1) Assign every sample to its nearest centroid.
		idx_t changed = 0;
		for (idx_t i = 0; i < n; i++) {
			float best = std::numeric_limits<float>::infinity();
			idx_t best_c = 0;
			for (idx_t c = 0; c < k; c++) {
				const float d = L2sq(data + i * dim, centroids_out + c * dim, dim);
				if (d < best) {
					best = d;
					best_c = c;
				}
			}
			if (assign[i] != best_c) {
				changed++;
				assign[i] = best_c;
			}
		}

		// 2) Recompute centroids as the mean of their assigned points.
		std::fill(sum_vec.begin(), sum_vec.end(), 0.0);
		std::fill(count.begin(), count.end(), 0);
		for (idx_t i = 0; i < n; i++) {
			const idx_t c = assign[i];
			count[c]++;
			const float *v = data + i * dim;
			double *acc = sum_vec.data() + c * dim;
			for (idx_t d = 0; d < dim; d++) {
				acc[d] += v[d];
			}
		}
		for (idx_t c = 0; c < k; c++) {
			if (count[c] > 0) {
				double inv = 1.0 / double(count[c]);
				float *cen = centroids_out + c * dim;
				double *acc = sum_vec.data() + c * dim;
				for (idx_t d = 0; d < dim; d++) {
					cen[d] = float(acc[d] * inv);
				}
			} else {
				// Empty cluster: reseed from the point farthest from its
				// assigned centroid. Cheap correction — matters on skewed
				// distributions where some seed was a near-duplicate.
				float worst = -1.0f;
				idx_t worst_i = 0;
				for (idx_t i = 0; i < n; i++) {
					const float d = L2sq(data + i * dim, centroids_out + assign[i] * dim, dim);
					if (d > worst) {
						worst = d;
						worst_i = i;
					}
				}
				std::memcpy(centroids_out + c * dim, data + worst_i * dim, dim * sizeof(float));
			}
		}

		// Early stop: <1% of points changed assignment.
		if (iter > 0 && changed * 100 < n) {
			break;
		}
	}
}

} // namespace ivf
} // namespace vindex
} // namespace duckdb
