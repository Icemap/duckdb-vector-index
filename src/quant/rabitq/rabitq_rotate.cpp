#include "rabitq_rotate.hpp"

#include "duckdb/common/exception.hpp"

#include <cmath>
#include <cstring>
#include <random>

namespace duckdb {
namespace vindex {
namespace rabitq {

RandomRotation::RandomRotation(idx_t dim, uint64_t seed) : dim_(dim), seed_(seed) {
	Rebuild();
}

void RandomRotation::Rebuild() {
	if (dim_ == 0) {
		throw InternalException("RandomRotation: dim must be positive");
	}
	// Gaussian matrix → Modified Gram-Schmidt → orthogonal rotation.
	// std::mt19937_64 + std::normal_distribution are deterministic across
	// libstdc++/libc++ for a fixed sequence of calls, which is what we need
	// for persistence round-trip.
	std::mt19937_64 rng(seed_);
	std::normal_distribution<float> nd(0.0f, 1.0f);

	matrix_.assign(dim_ * dim_, 0.0f);
	for (idx_t i = 0; i < dim_; i++) {
		for (idx_t j = 0; j < dim_; j++) {
			matrix_[i * dim_ + j] = nd(rng);
		}
	}

	// Modified Gram-Schmidt orthonormalization. Row i is orthogonalized against
	// all rows 0..i-1, then normalized.
	for (idx_t i = 0; i < dim_; i++) {
		float *row_i = matrix_.data() + i * dim_;
		for (idx_t j = 0; j < i; j++) {
			const float *row_j = matrix_.data() + j * dim_;
			float dot = 0.0f;
			for (idx_t k = 0; k < dim_; k++) {
				dot += row_i[k] * row_j[k];
			}
			for (idx_t k = 0; k < dim_; k++) {
				row_i[k] -= dot * row_j[k];
			}
		}
		float norm_sq = 0.0f;
		for (idx_t k = 0; k < dim_; k++) {
			norm_sq += row_i[k] * row_i[k];
		}
		// A near-zero norm means the Gaussian draw was (with astronomically
		// small probability) in the span of earlier rows. Bailing here makes
		// the failure loud instead of silently producing a degenerate matrix.
		if (norm_sq < 1e-20f) {
			throw InternalException("RandomRotation: row %llu collapsed during MGS (seed=%llu, dim=%llu)",
			                        (unsigned long long)i, (unsigned long long)seed_, (unsigned long long)dim_);
		}
		const float inv = 1.0f / std::sqrt(norm_sq);
		for (idx_t k = 0; k < dim_; k++) {
			row_i[k] *= inv;
		}
	}
}

void RandomRotation::Apply(const float *x, float *out) const {
	// out[i] = sum_k R[i,k] * x[k]
	for (idx_t i = 0; i < dim_; i++) {
		const float *row = matrix_.data() + i * dim_;
		float acc = 0.0f;
		for (idx_t k = 0; k < dim_; k++) {
			acc += row[k] * x[k];
		}
		out[i] = acc;
	}
}

void RandomRotation::ApplyTranspose(const float *x, float *out) const {
	// out[i] = sum_k R[k,i] * x[k]
	for (idx_t i = 0; i < dim_; i++) {
		out[i] = 0.0f;
	}
	for (idx_t k = 0; k < dim_; k++) {
		const float *row = matrix_.data() + k * dim_;
		const float xk = x[k];
		for (idx_t i = 0; i < dim_; i++) {
			out[i] += row[i] * xk;
		}
	}
}

void RandomRotation::Serialize(vector<data_t> &out) const {
	out.resize(sizeof(uint64_t) * 2);
	const uint64_t d = dim_;
	std::memcpy(out.data(), &d, sizeof(d));
	std::memcpy(out.data() + sizeof(uint64_t), &seed_, sizeof(seed_));
}

void RandomRotation::Deserialize(const_data_ptr_t in, idx_t size) {
	if (size < sizeof(uint64_t) * 2) {
		throw InternalException("RandomRotation::Deserialize: blob too small (%llu)", (unsigned long long)size);
	}
	uint64_t d;
	std::memcpy(&d, in, sizeof(d));
	std::memcpy(&seed_, in + sizeof(uint64_t), sizeof(seed_));
	dim_ = d;
	Rebuild();
}

} // namespace rabitq
} // namespace vindex
} // namespace duckdb
