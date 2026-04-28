#pragma once

#include "duckdb/common/typedefs.hpp"

#include <cstdint>

namespace duckdb {
namespace vindex {
namespace ivf {

// k-means++ seeding + Lloyd iterations. Euclidean (L2) metric.
//
// Inputs:
//   data          (n × dim) float32, row-major
//   n             number of input samples (must be ≥ k)
//   dim           vector dimension
//   k             number of centroids
//   seed          RNG seed
//   max_iters     maximum Lloyd iterations. Terminates early when <1% of
//                 points change assignment between iterations.
//
// Output:
//   centroids_out (k × dim) float32, row-major, preallocated by caller.
//
// Notes:
//   - If n < k, the function still returns k centroids — extra ones are
//     seeded by repeating random samples (caller should ensure n >= k in
//     production; this is defensive).
//   - Empty clusters (no point assigned) are reseeded from the sample with
//     the largest distance to its current centroid.
void KMeansPlusPlus(const float *data, idx_t n, idx_t dim, idx_t k, uint64_t seed, idx_t max_iters,
                    float *centroids_out);

} // namespace ivf
} // namespace vindex
} // namespace duckdb
