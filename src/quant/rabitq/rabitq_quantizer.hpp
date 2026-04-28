#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

#include "rabitq_rotate.hpp"

namespace duckdb {
namespace vindex {
namespace rabitq {

// ---------------------------------------------------------------------------
// RabitqQuantizer — b-bit scalar reference (ExRaBitQ, task #18).
//
// References:
//   Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
//   Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024.
//
// Supported bit widths: 1, 2, 3, 4, 5, 7, 8. Default is 3 (see README.md
// §"Quantizer bits vs recall"). bits=6 is intentionally omitted — it is
// pareto-dominated by bits=5 + bigger rerank on every workload we care about.
//
// Per-vector storage layout (CodeSize = ceil(dim*bits/8) + 12):
//   [0 .. ceil(dim*bits/8) - 1]   packed b-bit codes, LSB-first, dim 0 lowest
//   [..+4]                        α  = ‖x_eff - c‖                 (float32)
//   [..+4]                        scale                            (float32)
//     ├─ bits == 1:  scale = β  = <o_bar, b_bar>   (RaBitQ 1-bit calibration)
//     └─ bits >= 2:  scale = Δ  = max|r_rot| / (2^(bits-1) - 0.5)
//                                 (per-vector uniform scalar quant step)
//   [..+4]                        rc = <r, c> = <x_eff - c, c>     (float32)
//                                 L2SQ ignores it; IP/COSINE use it to fold
//                                 the residual dot back into the full
//                                 inner product.
//
// `x_eff = x` for L2SQ / IP and `x_eff = x / ‖x‖` for COSINE, so COSINE is
// implemented as IP on the unit sphere.
//
// Query workspace (QueryWorkspaceSize = dim + 2 floats):
//   [0 .. dim-1]  q_rot = R · (q_eff - c)
//   [dim]         ‖q_eff - c‖²      (L2SQ)
//   [dim+1]       <c, q_eff>        (IP / COSINE)
//
// Estimators (`cross` ≈ <r_rot, q_rot>):
//   bits == 1:  cross = α · Σ(b̄·q_rot) / β
//               where b̄[i] = (2·bit[i] - 1)/√d maps {0,1} → {-1,+1}/√d.
//   bits >= 2:  cross = Δ · Σ code[i]·q_rot[i]
//               where code[i] ∈ {-2^(b-1), ..., 2^(b-1)-1} is the signed
//               per-dim scalar quant of r_rot, stored as two's complement.
//
//   L2SQ:    ‖x - q‖²        ≈ α² + ‖q-c‖² - 2·cross
//   IP:      -<x, q>         ≈ -(cross + rc + <c, q>)
//   COSINE:  1 - cos(x, q)   ≈ 1 - (cross + rc + <c, q̂>)
// ---------------------------------------------------------------------------

class RabitqQuantizer : public Quantizer {
public:
	RabitqQuantizer(MetricKind metric, idx_t dim, uint8_t bits = 3, uint64_t rotation_seed = 0xB17C0DE);

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
		return QuantizerKind::RABITQ;
	}

	void Serialize(vector<data_t> &out) const override;
	void Deserialize(const_data_ptr_t in, idx_t size) override;

	// Accessors for tests.
	const vector<float> &Centroid() const {
		return centroid_;
	}
	const RandomRotation &Rotation() const {
		return rotation_;
	}
	bool IsTrained() const {
		return trained_;
	}

	uint8_t Bits() const {
		return bits_;
	}

private:
	// Number of bytes needed to pack `dim_ * bits_` bits.
	idx_t BitBytes() const {
		return (dim_ * bits_ + 7) / 8;
	}

	MetricKind metric_;
	idx_t dim_;
	uint8_t bits_;
	uint64_t rotation_seed_;
	bool trained_ = false;
	vector<float> centroid_;
	float centroid_norm_sq_ = 0.0f; // cached ‖c‖² for the IP / COSINE CodeDistance path
	RandomRotation rotation_;
};

} // namespace rabitq
} // namespace vindex
} // namespace duckdb
