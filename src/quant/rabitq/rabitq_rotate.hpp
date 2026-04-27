#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace vindex {
namespace rabitq {

// ---------------------------------------------------------------------------
// RandomRotation — deterministic orthogonal d×d matrix used as RaBitQ's
// pre-quantization rotation. Applying R to a vector x produces Rx whose
// coordinates are approximately iid (for the N(0,1) → QR construction), which
// is what makes the per-coordinate bit-quantization error well-behaved.
//
// The matrix is fully determined by (dim, seed): persistence just stores the
// seed plus dim, and Deserialize rebuilds the matrix. This keeps index blobs
// small — for dim=1024, storing the matrix verbatim would be 4MB; storing the
// seed is 16 bytes.
// ---------------------------------------------------------------------------
class RandomRotation {
public:
	RandomRotation() = default;
	RandomRotation(idx_t dim, uint64_t seed);

	// out = R * x. `x` and `out` must be distinct buffers of length Dim().
	void Apply(const float *x, float *out) const;

	// out = R^T * x. Since R is orthogonal, this is the inverse rotation.
	void ApplyTranspose(const float *x, float *out) const;

	idx_t Dim() const {
		return dim_;
	}
	uint64_t Seed() const {
		return seed_;
	}

	// Serialization layout: {dim:u64, seed:u64}. The matrix itself is
	// reconstructed deterministically on Deserialize.
	void Serialize(vector<data_t> &out) const;
	void Deserialize(const_data_ptr_t in, idx_t size);

private:
	void Rebuild();

	idx_t dim_ = 0;
	uint64_t seed_ = 0;
	vector<float> matrix_; // row-major, dim_ × dim_
};

} // namespace rabitq
} // namespace vindex
} // namespace duckdb
