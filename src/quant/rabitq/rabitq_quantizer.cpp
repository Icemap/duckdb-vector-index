#include "rabitq_quantizer.hpp"

#include "rabitq_kernels.hpp"

#include "duckdb/common/exception.hpp"

#include <cmath>
#include <cstring>

namespace duckdb {
namespace vindex {
namespace rabitq {

// Per-vector trailer layout right after the packed bits.
//
//   bits == 1: (alpha = ‖x_eff - c‖,  scale = β = <o_bar, b_bar>)
//   bits >= 2: (alpha = ‖x_eff - c‖,  scale = Δ = max|r_rot| / (2^(bits-1) - 0.5))
//
// `rc` caches <r, c> = <x_eff - c, c> per vector. L2SQ ignores it; IP and
// COSINE use it to reconstruct the full inner product as
//   <x_eff, q_eff> = cross + rc + <c, q_eff>
// where cross ≈ <r_rot, q_rot> is the estimator's residual dot. Storing
// this pre-computed scalar costs 4 bytes per code but avoids running the
// Dot kernel a second time against R·c per candidate.
//
// `x_eff = x` for L2SQ/IP and `x_eff = x/‖x‖` for COSINE.
struct CodeTrailer {
	float alpha;
	float scale;
	float rc;
};

namespace {

// ExRaBitQ supports b ∈ {1,2,3,4,5,7,8}. b=6 is omitted because rerank ×
// bits=5 dominates b=6 on every recall/byte trade-off we measured; see README
// §"Quantizer bits vs recall".
bool IsSupportedBits(uint8_t bits) {
	switch (bits) {
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 7:
	case 8:
		return true;
	default:
		return false;
	}
}

// Pack `dim` codes of `bits` bits each into `out`, LSB-first, dim 0 first.
// `signed_codes[i]` is a signed integer in [-2^(b-1), 2^(b-1)-1]; it is stored
// as two's complement truncated to `bits` low bits.
void PackBits(const int32_t *signed_codes, idx_t dim, uint8_t bits, data_ptr_t out) {
	const idx_t total_bits = dim * bits;
	const idx_t byte_count = (total_bits + 7) / 8;
	std::memset(out, 0, byte_count);
	const uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
	for (idx_t i = 0; i < dim; i++) {
		const uint32_t code = static_cast<uint32_t>(signed_codes[i]) & mask;
		const idx_t bit_off = i * bits;
		const idx_t byte_off = bit_off / 8;
		const uint8_t shift = static_cast<uint8_t>(bit_off % 8);
		// bits ≤ 8 ⇒ the code straddles at most two bytes.
		out[byte_off] |= static_cast<uint8_t>((code << shift) & 0xFFu);
		const uint8_t used = static_cast<uint8_t>(8 - shift);
		if (used < bits) {
			out[byte_off + 1] |= static_cast<uint8_t>((code >> used) & 0xFFu);
		}
	}
}

// Inverse of PackBits — returns a signed int in [-2^(b-1), 2^(b-1)-1].
int32_t UnpackBits(const_data_ptr_t in, idx_t i, uint8_t bits) {
	const idx_t bit_off = i * bits;
	const idx_t byte_off = bit_off / 8;
	const uint8_t shift = static_cast<uint8_t>(bit_off % 8);
	const uint32_t mask = (1u << bits) - 1u;
	uint32_t raw = static_cast<uint32_t>(in[byte_off]) >> shift;
	const uint8_t used = static_cast<uint8_t>(8 - shift);
	if (used < bits) {
		raw |= static_cast<uint32_t>(in[byte_off + 1]) << used;
	}
	raw &= mask;
	// Sign-extend from `bits` to 32.
	const uint32_t sign_bit = 1u << (bits - 1);
	if (raw & sign_bit) {
		raw |= ~mask;
	}
	return static_cast<int32_t>(raw);
}

} // namespace

RabitqQuantizer::RabitqQuantizer(MetricKind metric, idx_t dim, uint8_t bits, uint64_t rotation_seed)
    : metric_(metric), dim_(dim), bits_(bits), rotation_seed_(rotation_seed), centroid_(dim, 0.0f) {
	if (dim_ == 0) {
		throw InternalException("RabitqQuantizer: dim must be positive");
	}
	if (!IsSupportedBits(bits_)) {
		throw NotImplementedException(
		    "vindex::RabitqQuantizer: bits=%d is not supported (expected one of 1,2,3,4,5,7,8)", int(bits_));
	}
}

namespace {

// Normalize `v` (length `dim`) to unit norm, writing to `out`. Returns the
// original norm. For zero vectors, emits zeros and returns 0.
float Normalize(const float *v, idx_t dim, float *out) {
	float norm_sq = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		norm_sq += v[i] * v[i];
	}
	const float norm = std::sqrt(norm_sq);
	if (norm <= 0.0f) {
		for (idx_t i = 0; i < dim; i++) {
			out[i] = 0.0f;
		}
		return 0.0f;
	}
	const float inv = 1.0f / norm;
	for (idx_t i = 0; i < dim; i++) {
		out[i] = v[i] * inv;
	}
	return norm;
}

} // namespace

void RabitqQuantizer::Train(const float *samples, idx_t n, idx_t dim) {
	if (dim != dim_) {
		throw InternalException("RabitqQuantizer::Train: dim mismatch (expected %llu, got %llu)",
		                        (unsigned long long)dim_, (unsigned long long)dim);
	}
	// centroid ← mean of the training sample (the classic "remove the mean"
	// trick that makes RaBitQ error bounds data-independent). For COSINE we
	// work in the unit-sphere space, so the centroid is the mean of the
	// L2-normalized samples.
	centroid_.assign(dim_, 0.0f);
	if (n > 0) {
		vector<float> tmp;
		if (metric_ == MetricKind::COSINE) {
			tmp.resize(dim_);
		}
		for (idx_t i = 0; i < n; i++) {
			const float *row = samples + i * dim_;
			if (metric_ == MetricKind::COSINE) {
				Normalize(row, dim_, tmp.data());
				row = tmp.data();
			}
			for (idx_t k = 0; k < dim_; k++) {
				centroid_[k] += row[k];
			}
		}
		const float inv_n = 1.0f / static_cast<float>(n);
		for (idx_t k = 0; k < dim_; k++) {
			centroid_[k] *= inv_n;
		}
	}
	centroid_norm_sq_ = 0.0f;
	for (idx_t k = 0; k < dim_; k++) {
		centroid_norm_sq_ += centroid_[k] * centroid_[k];
	}
	rotation_ = RandomRotation(dim_, rotation_seed_);
	trained_ = true;
}

idx_t RabitqQuantizer::CodeSize() const {
	return BitBytes() + sizeof(CodeTrailer);
}

idx_t RabitqQuantizer::QueryWorkspaceSize() const {
	// Layout:
	//   [0 .. dim-1]   q_rot = R · (q_eff - c)
	//   [dim]          ‖q_eff - c‖²          (L2SQ estimator)
	//   [dim+1]        <c, q_eff>            (IP / COSINE estimators)
	// where q_eff = query for L2SQ / IP, and q_eff = query/‖query‖ for COSINE.
	return dim_ + 2;
}

void RabitqQuantizer::Encode(const float *vec, data_ptr_t code_out) const {
	if (!trained_) {
		throw InternalException("RabitqQuantizer::Encode called before Train");
	}

	// For COSINE we quantize the unit-length x̂ = x/‖x‖, so <x̂, q̂> = cos(x,q).
	// Other metrics encode the raw vector directly.
	vector<float> x_eff_storage;
	const float *x_eff = vec;
	if (metric_ == MetricKind::COSINE) {
		x_eff_storage.resize(dim_);
		Normalize(vec, dim_, x_eff_storage.data());
		x_eff = x_eff_storage.data();
	}

	// r = x_eff - c, and r_rot = R · r.
	vector<float> r(dim_);
	float rc = 0.0f;
	for (idx_t i = 0; i < dim_; i++) {
		r[i] = x_eff[i] - centroid_[i];
		rc += r[i] * centroid_[i];
	}
	vector<float> r_rot(dim_);
	rotation_.Apply(r.data(), r_rot.data());

	// alpha = ‖r‖ (same as ‖r_rot‖ since R is orthogonal).
	float norm_sq = 0.0f;
	for (idx_t i = 0; i < dim_; i++) {
		norm_sq += r_rot[i] * r_rot[i];
	}
	const float alpha = std::sqrt(norm_sq);

	const idx_t bit_bytes = BitBytes();

	if (bits_ == 1) {
		// 1-bit RaBitQ: sign-pack + β calibration.
		std::memset(code_out, 0, bit_bytes);
		for (idx_t i = 0; i < dim_; i++) {
			if (r_rot[i] > 0.0f) {
				code_out[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
			}
		}

		// β = <o_bar, b_bar> where
		//   o_bar[i] = r_rot[i] / alpha          (unit vector in the rotated frame)
		//   b_bar[i] = (2·bit[i] - 1) / sqrt(d)
		float beta = 0.0f;
		if (alpha > 0.0f) {
			const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim_));
			const float inv_alpha = 1.0f / alpha;
			for (idx_t i = 0; i < dim_; i++) {
				const float b_bar = (r_rot[i] > 0.0f ? 1.0f : -1.0f) * inv_sqrt_d;
				const float o_bar = r_rot[i] * inv_alpha;
				beta += o_bar * b_bar;
			}
		}

		CodeTrailer trailer {alpha, beta, rc};
		std::memcpy(code_out + bit_bytes, &trailer, sizeof(trailer));
		return;
	}

	// bits ≥ 2: signed uniform scalar quant. The asymmetric range ±(2^(b-1) - 0.5)
	// splits the max-|r_rot| interval into 2^b equal-width bins centered on 0,
	// giving the same error as symmetric round-to-nearest but with the full
	// two's-complement range available.
	float max_abs = 0.0f;
	for (idx_t i = 0; i < dim_; i++) {
		const float a = std::fabs(r_rot[i]);
		if (a > max_abs) {
			max_abs = a;
		}
	}
	const float levels = static_cast<float>((1u << (bits_ - 1)) - 1) + 0.5f; // 2^(b-1) - 0.5
	const float delta = max_abs > 0.0f ? max_abs / levels : 0.0f;
	const float inv_delta = delta > 0.0f ? 1.0f / delta : 0.0f;

	const int32_t lo = -static_cast<int32_t>(1u << (bits_ - 1));
	const int32_t hi = static_cast<int32_t>((1u << (bits_ - 1)) - 1);

	vector<int32_t> codes(dim_);
	for (idx_t i = 0; i < dim_; i++) {
		int32_t q = static_cast<int32_t>(std::lrint(r_rot[i] * inv_delta));
		if (q < lo) {
			q = lo;
		} else if (q > hi) {
			q = hi;
		}
		codes[i] = q;
	}
	PackBits(codes.data(), dim_, bits_, code_out);

	CodeTrailer trailer {alpha, delta, rc};
	std::memcpy(code_out + bit_bytes, &trailer, sizeof(trailer));
}

void RabitqQuantizer::PreprocessQuery(const float *query, float *out) const {
	if (!trained_) {
		throw InternalException("RabitqQuantizer::PreprocessQuery called before Train");
	}

	// For COSINE we normalize the query once so the estimator computes
	// <x̂, q̂> directly. For IP/L2SQ we work in the raw coordinate system.
	vector<float> q_eff_storage;
	const float *q_eff = query;
	if (metric_ == MetricKind::COSINE) {
		q_eff_storage.resize(dim_);
		Normalize(query, dim_, q_eff_storage.data());
		q_eff = q_eff_storage.data();
	}

	// q_c = q_eff - centroid, then q_rot = R · q_c.
	// Side computations:
	//   ‖q_eff - c‖² — used by the L2SQ estimator.
	//   <c, q_eff>   — used by the IP / COSINE estimators to reconstruct
	//                  the full inner product from the residual dot.
	vector<float> q_c(dim_);
	float norm_sq = 0.0f;
	float c_dot_q = 0.0f;
	for (idx_t i = 0; i < dim_; i++) {
		q_c[i] = q_eff[i] - centroid_[i];
		norm_sq += q_c[i] * q_c[i];
		c_dot_q += centroid_[i] * q_eff[i];
	}
	rotation_.Apply(q_c.data(), out);
	out[dim_] = norm_sq;
	out[dim_ + 1] = c_dot_q;
}

float RabitqQuantizer::EstimateDistance(const_data_ptr_t code, const float *query_preproc) const {
	if (!trained_) {
		throw InternalException("RabitqQuantizer::EstimateDistance called before Train");
	}
	const idx_t bit_bytes = BitBytes();
	CodeTrailer trailer;
	std::memcpy(&trailer, code + bit_bytes, sizeof(trailer));
	const float alpha = trailer.alpha;
	const float scale = trailer.scale;
	const float rc = trailer.rc;
	const float q_norm_sq = query_preproc[dim_];
	const float c_dot_q = query_preproc[dim_ + 1];

	// Heavy lifting lives in rabitq_kernels.cpp so the SIMD hot paths are in
	// one translation unit, with a per-bits dispatch that the compiler can
	// specialize. `dot` has per-bits semantics: for bits=1 it is Σ(2b-1)·q (no
	// 1/√d yet), for bits≥2 it is Σ signed_code · q.
	const float dot = kernels::Dot(code, query_preproc, dim_, bits_);

	float cross = 0.0f;
	if (bits_ == 1) {
		// RaBitQ estimates <r_rot, q_rot> ≈ α · (dot/√d) / β. When β is 0
		// (degenerate all-zero vector) the unbiased fallback is to drop the
		// cross term so the estimate collapses to α² + ‖q-c‖².
		const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim_));
		cross = (scale > 0.0f) ? (alpha * dot * inv_sqrt_d / scale) : 0.0f;
	} else {
		// bits ≥ 2: r_rot[i] ≈ Δ · code[i] ⇒ <r_rot, q_rot> ≈ Δ · dot.
		cross = scale * dot;
	}

	switch (metric_) {
	case MetricKind::L2SQ: {
		// ‖x - q‖² = ‖r‖² + ‖q - c‖² - 2·<r_rot, q_rot>
		//          = α² + q_norm_sq - 2·cross
		const float est = alpha * alpha + q_norm_sq - 2.0f * cross;
		// Numerical floor — a zero-distance match can dip slightly below zero
		// due to float32 cancellation, but L2SQ is nonnegative by definition.
		return est < 0.0f ? 0.0f : est;
	}
	case MetricKind::IP: {
		// <x, q> = <r_rot, q_rot> + <r, c> + <c, q>
		// DuckDB's array_negative_inner_product convention negates so that
		// smaller = closer, matching the Flat quantizer's IP branch.
		return -(cross + rc + c_dot_q);
	}
	case MetricKind::COSINE: {
		// q_preproc was derived from q̂ = q/‖q‖ and the code from x̂ = x/‖x‖,
		// so `cross + rc + c_dot_q` ≈ <x̂, q̂> = cos(x, q). Distance is
		// 1 - cos to match Flat's cosine branch.
		return 1.0f - (cross + rc + c_dot_q);
	}
	}
	throw InternalException("RabitqQuantizer::EstimateDistance: unreachable MetricKind");
}

// Code-to-code distance used by the HNSW Algorithm 4 neighbor-pruning
// heuristic at insert time. The rotation R is orthogonal, so residual inner
// products and L2SQ distances in the rotated frame equal those in the
// original frame. We reconstruct r_rot_a, r_rot_b from each code and fold
// in the cached `rc` scalars to recover either ‖x_a - x_b‖² (L2SQ) or
// <x_eff_a, x_eff_b> (IP / COSINE). O(dim) per call, paid once per candidate
// edge during Insert().
float RabitqQuantizer::CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const {
	if (!trained_) {
		throw InternalException("RabitqQuantizer::CodeDistance called before Train");
	}
	const idx_t bit_bytes = BitBytes();
	CodeTrailer ta, tb;
	std::memcpy(&ta, code_a + bit_bytes, sizeof(ta));
	std::memcpy(&tb, code_b + bit_bytes, sizeof(tb));

	float l2sq_acc = 0.0f;
	float ip_acc = 0.0f; // <r_rot_a, r_rot_b> accumulator

	if (bits_ == 1) {
		// b̄[i] = (2·bit[i] - 1) / √d, r̂ ≈ α · b̄ / β, so
		//   r_a[i] ≈ (α_a/β_a)·(2·bit_a[i]-1)/√d.
		const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim_));
		const float sa = (ta.scale != 0.0f) ? ta.alpha / ta.scale * inv_sqrt_d : 0.0f;
		const float sb = (tb.scale != 0.0f) ? tb.alpha / tb.scale * inv_sqrt_d : 0.0f;
		for (idx_t i = 0; i < dim_; i++) {
			const int bit_a = int(UnpackBits(code_a, i, 1)) & 1;
			const int bit_b = int(UnpackBits(code_b, i, 1)) & 1;
			const float ra = sa * (2.0f * bit_a - 1.0f);
			const float rb = sb * (2.0f * bit_b - 1.0f);
			const float d = ra - rb;
			l2sq_acc += d * d;
			ip_acc += ra * rb;
		}
	} else {
		// bits ≥ 2: r_rot[i] ≈ Δ · code[i]. The trailer's `scale` is the
		// per-vector Δ already, so just subtract the scaled signed codes.
		for (idx_t i = 0; i < dim_; i++) {
			const int32_t ca = UnpackBits(code_a, i, bits_);
			const int32_t cb = UnpackBits(code_b, i, bits_);
			const float ra = ta.scale * float(ca);
			const float rb = tb.scale * float(cb);
			const float d = ra - rb;
			l2sq_acc += d * d;
			ip_acc += ra * rb;
		}
	}

	switch (metric_) {
	case MetricKind::L2SQ:
		return l2sq_acc;
	case MetricKind::IP:
	case MetricKind::COSINE: {
		// <x_eff_a, x_eff_b> = <r_a, r_b> + <r_a, c> + <r_b, c> + ‖c‖²
		// ‖c‖² is constant across all pairs but included so the returned
		// scalar matches the estimator's semantics (negated IP, 1 - cos).
		const float dot = ip_acc + ta.rc + tb.rc + centroid_norm_sq_;
		return metric_ == MetricKind::IP ? -dot : (1.0f - dot);
	}
	}
	throw InternalException("RabitqQuantizer::CodeDistance: unreachable MetricKind");
}

// Persistence layout:
//   magic:u8=0x52 ('R'), bits:u8, metric:u8, pad:u8, dim:u64,
//   rotation_seed:u64, centroid[dim]:float, rotation_blob:{u64 size, bytes}.
void RabitqQuantizer::Serialize(vector<data_t> &out) const {
	if (!trained_) {
		throw InternalException("RabitqQuantizer::Serialize called before Train");
	}
	vector<data_t> rot_blob;
	rotation_.Serialize(rot_blob);

	const idx_t header_bytes = 4 + sizeof(uint64_t) + sizeof(uint64_t);
	const idx_t centroid_bytes = dim_ * sizeof(float);
	const idx_t rot_prefix = sizeof(uint64_t);
	out.assign(header_bytes + centroid_bytes + rot_prefix + rot_blob.size(), 0);

	auto ptr = out.data();
	ptr[0] = 0x52;
	ptr[1] = bits_;
	ptr[2] = static_cast<uint8_t>(metric_);
	ptr[3] = 0;
	ptr += 4;

	const uint64_t d = dim_;
	std::memcpy(ptr, &d, sizeof(d));
	ptr += sizeof(d);
	std::memcpy(ptr, &rotation_seed_, sizeof(rotation_seed_));
	ptr += sizeof(rotation_seed_);
	std::memcpy(ptr, centroid_.data(), centroid_bytes);
	ptr += centroid_bytes;

	const uint64_t rot_size = rot_blob.size();
	std::memcpy(ptr, &rot_size, sizeof(rot_size));
	ptr += sizeof(rot_size);
	std::memcpy(ptr, rot_blob.data(), rot_blob.size());
}

void RabitqQuantizer::Deserialize(const_data_ptr_t in, idx_t size) {
	const idx_t header_bytes = 4 + sizeof(uint64_t) + sizeof(uint64_t);
	if (size < header_bytes) {
		throw InternalException("RabitqQuantizer::Deserialize: blob too small (%llu)", (unsigned long long)size);
	}
	if (in[0] != 0x52) {
		throw InternalException("RabitqQuantizer::Deserialize: bad magic 0x%02x", int(in[0]));
	}
	const uint8_t disk_bits = in[1];
	if (!IsSupportedBits(disk_bits)) {
		throw InternalException("RabitqQuantizer::Deserialize: unsupported bits=%d", int(disk_bits));
	}
	bits_ = disk_bits;
	metric_ = static_cast<MetricKind>(in[2]);
	auto ptr = in + 4;

	uint64_t d;
	std::memcpy(&d, ptr, sizeof(d));
	ptr += sizeof(d);
	std::memcpy(&rotation_seed_, ptr, sizeof(rotation_seed_));
	ptr += sizeof(rotation_seed_);
	dim_ = d;

	const idx_t centroid_bytes = dim_ * sizeof(float);
	if (static_cast<idx_t>(ptr - in) + centroid_bytes + sizeof(uint64_t) > size) {
		throw InternalException("RabitqQuantizer::Deserialize: centroid bytes out of range");
	}
	centroid_.assign(dim_, 0.0f);
	std::memcpy(centroid_.data(), ptr, centroid_bytes);
	ptr += centroid_bytes;
	centroid_norm_sq_ = 0.0f;
	for (idx_t k = 0; k < dim_; k++) {
		centroid_norm_sq_ += centroid_[k] * centroid_[k];
	}

	uint64_t rot_size;
	std::memcpy(&rot_size, ptr, sizeof(rot_size));
	ptr += sizeof(rot_size);
	if (static_cast<idx_t>(ptr - in) + rot_size > size) {
		throw InternalException("RabitqQuantizer::Deserialize: rotation blob out of range");
	}
	rotation_.Deserialize(ptr, rot_size);
	trained_ = true;
}

} // namespace rabitq
} // namespace vindex
} // namespace duckdb
