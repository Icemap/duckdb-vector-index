#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

namespace duckdb {
namespace vindex {
namespace scann {

// ---------------------------------------------------------------------------
// ScannQuantizer — anisotropic product quantization (Guo et al., ICML 2020).
//
// Same layout as PQ (sub-vector split, per-slot codebook of 2^bits centroids,
// AH distance table). The difference is in how centroids are *trained* and
// how a sub-vector is *assigned* to a centroid.
//
// PQ's Lloyd iteration minimises the unweighted L2 reconstruction error:
//       Σ_i ||x_i - c(x_i)||²
// ScaNN observes that for inner-product / cosine retrieval, the error
// component that is *parallel* to the vector hurts score ranking much more
// than the perpendicular component. Reformulate the loss as
//       Σ_i  h_parallel    · ||e_parallel(x_i)||²
//     +      h_perpendicular · ||e_perpendicular(x_i)||²
// with h_parallel > h_perpendicular (default h_parallel / h_perpendicular = 4,
// ICML 2020 §5). We solve this with a weighted Lloyd: each point contributes
// to the centroid update with weight w_i = h_parallel · r²_∥ + h_perp · r²_⊥
// where r is the residual x_i - c(x_i), and projection is taken against
// x_i itself (so each sub-vector defines its own axis).
//
// At query time the math is identical to PQ — ADC table lookups sum across
// slots. The scoring creativity is entirely at *training* time.
//
// Defaults: bits=8, m=dim/4, eta=4.0 (h_parallel / h_perpendicular).
// L2SQ and IP are supported; COSINE is rejected (PQ-family quantizers need
// per-vector norm tracking for cosine; left out to keep the code minimal —
// use IP on pre-normalized vectors instead).
// ---------------------------------------------------------------------------

class ScannQuantizer : public Quantizer {
public:
	ScannQuantizer(MetricKind metric, idx_t dim, uint8_t m, uint8_t bits, float eta = 4.0f,
	               uint64_t seed = 0x5CA77EED1234ULL);

	void Train(const float *samples, idx_t n, idx_t dim) override;

	void Encode(const float *vec, data_ptr_t code_out) const override;
	float EstimateDistance(const_data_ptr_t code, const float *query_preproc) const override;
	float CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const override;
	void PreprocessQuery(const float *query, float *out) const override;

	idx_t CodeSize() const override;
	idx_t QueryWorkspaceSize() const override;
	MetricKind Metric() const override {
		return metric_;
	}
	QuantizerKind Kind() const override {
		return QuantizerKind::SCANN;
	}

	void Serialize(vector<data_t> &out) const override;
	void Deserialize(const_data_ptr_t in, idx_t size) override;

	// Introspection for tests / pragmas.
	uint8_t M() const {
		return m_;
	}
	uint8_t Bits() const {
		return bits_;
	}
	float Eta() const {
		return eta_;
	}

private:
	uint32_t ReadCode(const_data_ptr_t code, idx_t s) const;
	void WriteCode(data_ptr_t code, idx_t s, uint32_t cid) const;

	// Pick the nearest centroid under anisotropic loss against the full vector
	// `full` (needed for the parallel/perpendicular split — a sub-vector alone
	// doesn't define a direction). `sub` is the slot-s window of `full`.
	uint32_t NearestCentroidAniso(idx_t s, const float *sub, const float *full) const;

	// Lloyd iteration helper — runs one pass of weighted assign+update for
	// slot `s`. Returns the number of points whose assignment changed.
	idx_t AnisoLloydStep(idx_t s, const float *full_samples, const float *sub_samples, idx_t n, vector<idx_t> &assign);

	idx_t SubDim() const {
		return dim_ / idx_t(m_);
	}
	idx_t CentroidsPerSlot() const {
		return idx_t(1) << bits_;
	}

	MetricKind metric_;
	idx_t dim_ = 0;
	uint8_t m_ = 0;
	uint8_t bits_ = 0;
	float eta_ = 4.0f; // h_parallel / h_perpendicular
	uint64_t seed_;
	bool trained_ = false;

	// Flat codebook: for slot s, codebook_[s * (2^bits) * sub_dim .. ] holds
	// `2^bits` centroids of `sub_dim` floats each.
	vector<float> codebook_;
};

} // namespace scann
} // namespace vindex
} // namespace duckdb
