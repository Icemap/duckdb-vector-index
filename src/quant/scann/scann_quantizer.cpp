#include "vindex/scann_quantizer.hpp"

#include "algo/ivf/kmeans.hpp"

#include "duckdb/common/exception.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace duckdb {
namespace vindex {
namespace scann {

namespace {

inline float L2sq(const float *a, const float *b, idx_t dim) {
	simsimd_distance_t d = 0;
	simsimd_l2sq_f32(reinterpret_cast<const simsimd_f32_t *>(a), reinterpret_cast<const simsimd_f32_t *>(b), dim, &d);
	return float(d);
}

inline float Dot(const float *a, const float *b, idx_t dim) {
	simsimd_distance_t d = 0;
	simsimd_dot_f32(reinterpret_cast<const simsimd_f32_t *>(a), reinterpret_cast<const simsimd_f32_t *>(b), dim, &d);
	return float(d);
}

// Anisotropic loss contribution of residual `r` against anchor direction `x`:
//    w(r | x) = h_∥ · (r · x̂)²  +  h_⊥ · (||r||² - (r · x̂)²)
// where x̂ = x / ||x||. h_∥ > h_⊥ penalises error along x̂ more than perp to it.
// `eta` = h_∥ / h_⊥; we normalise h_⊥ = 1 without loss of generality.
inline float AnisoCost(const float *r, const float *x, idx_t dim, float eta) {
	float r_norm_sq = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		r_norm_sq += r[i] * r[i];
	}
	float x_norm_sq = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		x_norm_sq += x[i] * x[i];
	}
	if (x_norm_sq <= 1e-12f) {
		// Degenerate: x is (near-)zero, direction undefined. Fall back to
		// isotropic L2 — anisotropic loss collapses to ordinary squared error.
		return r_norm_sq;
	}
	float r_dot_x = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		r_dot_x += r[i] * x[i];
	}
	const float parallel_sq = (r_dot_x * r_dot_x) / x_norm_sq;
	const float perp_sq = r_norm_sq - parallel_sq;
	return eta * parallel_sq + perp_sq;
}

} // namespace

ScannQuantizer::ScannQuantizer(MetricKind metric, idx_t dim, uint8_t m, uint8_t bits, float eta, uint64_t seed)
    : metric_(metric), dim_(dim), m_(m), bits_(bits), eta_(eta), seed_(seed) {
	if (metric_ == MetricKind::COSINE) {
		throw InvalidInputException("vindex: ScaNN quantizer does not support cosine metric (use L2SQ or IP on "
		                            "pre-normalised vectors)");
	}
	if (m_ == 0) {
		throw InvalidInputException("vindex: ScaNN 'm' must be > 0");
	}
	if (dim_ % idx_t(m_) != 0) {
		throw InvalidInputException("vindex: ScaNN requires dim (%llu) to be divisible by m (%d)",
		                            (unsigned long long)dim_, int(m_));
	}
	if (bits_ != 4 && bits_ != 8) {
		throw InvalidInputException("vindex: ScaNN 'bits' must be 4 or 8 (got %d)", int(bits_));
	}
	if (!(eta_ > 0.0f)) {
		throw InvalidInputException("vindex: ScaNN 'eta' must be > 0 (got %f)", double(eta_));
	}
}

idx_t ScannQuantizer::CodeSize() const {
	return (idx_t(m_) * bits_ + 7) / 8;
}

idx_t ScannQuantizer::QueryWorkspaceSize() const {
	return idx_t(m_) * CentroidsPerSlot();
}

uint32_t ScannQuantizer::ReadCode(const_data_ptr_t code, idx_t s) const {
	if (bits_ == 8) {
		return static_cast<uint32_t>(code[s]);
	}
	const idx_t byte_off = s / 2;
	const uint8_t shift = (s % 2) * 4;
	return (static_cast<uint32_t>(code[byte_off]) >> shift) & 0x0Fu;
}

void ScannQuantizer::WriteCode(data_ptr_t code, idx_t s, uint32_t cid) const {
	if (bits_ == 8) {
		code[s] = static_cast<uint8_t>(cid & 0xFFu);
		return;
	}
	const idx_t byte_off = s / 2;
	const uint8_t shift = (s % 2) * 4;
	const uint8_t nibble = static_cast<uint8_t>(cid & 0x0Fu);
	code[byte_off] = static_cast<uint8_t>((code[byte_off] & ~(0x0Fu << shift)) | (nibble << shift));
}

uint32_t ScannQuantizer::NearestCentroidAniso(idx_t s, const float *sub, const float *full) const {
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	const float *slot_book = codebook_.data() + s * k * sub_dim;

	vector<float> residual(sub_dim);
	uint32_t best = 0;
	float best_cost = std::numeric_limits<float>::infinity();
	for (idx_t c = 0; c < k; c++) {
		const float *cen = slot_book + c * sub_dim;
		for (idx_t i = 0; i < sub_dim; i++) {
			residual[i] = sub[i] - cen[i];
		}
		// Anchor direction is the sub-vector itself, not the full vector —
		// keeping the parallel/perp decomposition slot-local is what makes
		// anisotropic PQ tractable (full-vector anchor would couple slots).
		const float cost = AnisoCost(residual.data(), sub, sub_dim, eta_);
		if (cost < best_cost) {
			best_cost = cost;
			best = uint32_t(c);
		}
	}
	(void)full; // reserved for full-vector anchor variants; slot-local anchor
	            // is the default from ICML 2020 §5.2.
	return best;
}

idx_t ScannQuantizer::AnisoLloydStep(idx_t s, const float *full_samples, const float *sub_samples, idx_t n,
                                     vector<idx_t> &assign) {
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	float *slot_book = codebook_.data() + s * k * sub_dim;

	// 1) Assign.
	idx_t changed = 0;
	for (idx_t i = 0; i < n; i++) {
		const uint32_t best = NearestCentroidAniso(s, sub_samples + i * sub_dim, full_samples + i * dim_);
		if (assign[i] != static_cast<idx_t>(best)) {
			changed++;
			assign[i] = best;
		}
	}

	// 2) Weighted update: for each cluster, the new centroid minimises
	//        Σ_i w_i · ||x_i - c||²
	// where the direction-aware weight w_i depends on (x_i - c_old). We use
	// the *residual against the current centroid* to compute w_i (fixed-point
	// iteration — the weights update with the centroids). This is the
	// standard implementation in google/scann's `AsymmetricHashing2` trainer.
	vector<double> sum_vec(k * sub_dim, 0.0);
	vector<double> weight(k, 0.0);
	vector<float> residual(sub_dim);
	for (idx_t i = 0; i < n; i++) {
		const idx_t c = assign[i];
		const float *sub = sub_samples + i * sub_dim;
		const float *cen = slot_book + c * sub_dim;
		for (idx_t d = 0; d < sub_dim; d++) {
			residual[d] = sub[d] - cen[d];
		}

		// Weight = anisotropic cost of this residual against its own sub-
		// vector direction. Points on strong directions (high ||x_sub||) pull
		// their centroid harder — exactly the ScaNN bias.
		double r_norm_sq = 0.0;
		double x_norm_sq = 0.0;
		double r_dot_x = 0.0;
		for (idx_t d = 0; d < sub_dim; d++) {
			r_norm_sq += double(residual[d]) * residual[d];
			x_norm_sq += double(sub[d]) * sub[d];
			r_dot_x += double(residual[d]) * sub[d];
		}
		double w;
		if (x_norm_sq <= 1e-12) {
			w = r_norm_sq; // isotropic fallback
		} else {
			const double parallel_sq = (r_dot_x * r_dot_x) / x_norm_sq;
			const double perp_sq = r_norm_sq - parallel_sq;
			w = double(eta_) * parallel_sq + perp_sq;
		}
		// Use a weight floor so a perfectly-quantised point (w=0) still
		// contributes to the centroid update. Without this, cluster centroids
		// on well-separated data lock in place after one iteration.
		w = std::max(w, 1e-6);

		weight[c] += w;
		double *acc = sum_vec.data() + c * sub_dim;
		for (idx_t d = 0; d < sub_dim; d++) {
			acc[d] += w * double(sub[d]);
		}
	}
	for (idx_t c = 0; c < k; c++) {
		if (weight[c] > 0.0) {
			const double inv = 1.0 / weight[c];
			float *cen = slot_book + c * sub_dim;
			const double *acc = sum_vec.data() + c * sub_dim;
			for (idx_t d = 0; d < sub_dim; d++) {
				cen[d] = float(acc[d] * inv);
			}
		} else {
			// Empty cluster — seed with the sample that carries the largest
			// reconstruction cost (same trick as kmeans.cpp:142).
			idx_t worst_i = 0;
			float worst = -1.0f;
			for (idx_t i = 0; i < n; i++) {
				const float *sub = sub_samples + i * sub_dim;
				const float *cen = slot_book + assign[i] * sub_dim;
				const float d2 = L2sq(sub, cen, sub_dim);
				if (d2 > worst) {
					worst = d2;
					worst_i = i;
				}
			}
			std::memcpy(slot_book + c * sub_dim, sub_samples + worst_i * sub_dim, sub_dim * sizeof(float));
		}
	}
	return changed;
}

void ScannQuantizer::Train(const float *samples, idx_t n, idx_t dim) {
	if (dim != dim_) {
		throw InternalException("ScannQuantizer::Train dim mismatch (%llu vs %llu)", (unsigned long long)dim,
		                        (unsigned long long)dim_);
	}
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	codebook_.assign(idx_t(m_) * k * sub_dim, 0.0f);

	const idx_t max_iters = 25;
	vector<float> sub_buffer(n * sub_dim);
	vector<idx_t> assign(n, std::numeric_limits<idx_t>::max()); // sentinel

	for (idx_t s = 0; s < idx_t(m_); s++) {
		// Extract slot-s column block.
		for (idx_t i = 0; i < n; i++) {
			std::memcpy(sub_buffer.data() + i * sub_dim, samples + i * dim + s * sub_dim, sub_dim * sizeof(float));
		}

		// Seed with kmeans++ on the sub-vector block (isotropic seeding is
		// fine — seeds only initialise the Lloyd iteration).
		const uint64_t slot_seed = seed_ ^ (0x9E3779B97F4A7C15ULL * (s + 1));
		ivf::KMeansPlusPlus(sub_buffer.data(), n, sub_dim, k, slot_seed, /*max_iters=*/1,
		                    codebook_.data() + s * k * sub_dim);

		// Re-initialise assignments to force the first AnisoLloydStep to
		// count every point as "changed".
		std::fill(assign.begin(), assign.end(), std::numeric_limits<idx_t>::max());

		// Run anisotropic-weighted Lloyd.
		for (idx_t iter = 0; iter < max_iters; iter++) {
			const idx_t changed = AnisoLloydStep(s, samples, sub_buffer.data(), n, assign);
			// Early stop: <1% of points changed assignment.
			if (iter > 0 && changed * 100 < n) {
				break;
			}
		}
	}
	trained_ = true;
}

void ScannQuantizer::Encode(const float *vec, data_ptr_t code_out) const {
	if (!trained_) {
		throw InternalException("ScannQuantizer::Encode called before Train");
	}
	const idx_t sub_dim = SubDim();
	std::memset(code_out, 0, CodeSize());
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t cid = NearestCentroidAniso(s, vec + s * sub_dim, vec);
		WriteCode(code_out, s, cid);
	}
}

void ScannQuantizer::PreprocessQuery(const float *query, float *out) const {
	if (!trained_) {
		throw InternalException("ScannQuantizer::PreprocessQuery called before Train");
	}
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const float *q_sub = query + s * sub_dim;
		const float *slot_book = codebook_.data() + s * k * sub_dim;
		float *row = out + s * k;
		for (idx_t c = 0; c < k; c++) {
			const float *cen = slot_book + c * sub_dim;
			switch (metric_) {
			case MetricKind::L2SQ:
				row[c] = L2sq(q_sub, cen, sub_dim);
				break;
			case MetricKind::IP:
				row[c] = -Dot(q_sub, cen, sub_dim);
				break;
			case MetricKind::COSINE:
				throw InternalException("ScannQuantizer: cosine unreachable");
			}
		}
	}
}

float ScannQuantizer::EstimateDistance(const_data_ptr_t code, const float *query_preproc) const {
	const idx_t k = CentroidsPerSlot();
	float acc = 0.0f;
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t cid = ReadCode(code, s);
		acc += query_preproc[s * k + cid];
	}
	return acc;
}

float ScannQuantizer::CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const {
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	float acc = 0.0f;
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t ca = ReadCode(code_a, s);
		const uint32_t cb = ReadCode(code_b, s);
		const float *book = codebook_.data() + s * k * sub_dim;
		switch (metric_) {
		case MetricKind::L2SQ:
			acc += L2sq(book + ca * sub_dim, book + cb * sub_dim, sub_dim);
			break;
		case MetricKind::IP:
			acc += -Dot(book + ca * sub_dim, book + cb * sub_dim, sub_dim);
			break;
		case MetricKind::COSINE:
			throw InternalException("ScannQuantizer: cosine unreachable");
		}
	}
	return acc;
}

void ScannQuantizer::Serialize(vector<data_t> &out) const {
	// {kind:u8, metric:u8, m:u8, bits:u8, eta:f32, dim:u64, codebook:float32[...]}
	const idx_t header = sizeof(uint8_t) * 4 + sizeof(float) + sizeof(uint64_t);
	const idx_t book_bytes = codebook_.size() * sizeof(float);
	out.resize(header + book_bytes);
	auto ptr = out.data();
	ptr[0] = static_cast<uint8_t>(QuantizerKind::SCANN);
	ptr[1] = static_cast<uint8_t>(metric_);
	ptr[2] = m_;
	ptr[3] = bits_;
	std::memcpy(ptr + 4, &eta_, sizeof(float));
	const uint64_t d = dim_;
	std::memcpy(ptr + 4 + sizeof(float), &d, sizeof(d));
	if (book_bytes > 0) {
		std::memcpy(ptr + header, codebook_.data(), book_bytes);
	}
}

void ScannQuantizer::Deserialize(const_data_ptr_t in, idx_t size) {
	const idx_t header = sizeof(uint8_t) * 4 + sizeof(float) + sizeof(uint64_t);
	if (size < header) {
		throw InternalException("ScannQuantizer::Deserialize: blob too small (%llu)", (unsigned long long)size);
	}
	const auto kind = static_cast<QuantizerKind>(in[0]);
	if (kind != QuantizerKind::SCANN) {
		throw InternalException("ScannQuantizer::Deserialize: wrong kind tag %d", int(in[0]));
	}
	metric_ = static_cast<MetricKind>(in[1]);
	m_ = in[2];
	bits_ = in[3];
	std::memcpy(&eta_, in + 4, sizeof(float));
	uint64_t d;
	std::memcpy(&d, in + 4 + sizeof(float), sizeof(d));
	dim_ = d;
	if (m_ == 0 || dim_ % idx_t(m_) != 0 || (bits_ != 4 && bits_ != 8) || !(eta_ > 0.0f)) {
		throw InternalException("ScannQuantizer::Deserialize: invalid header (m=%d bits=%d eta=%f dim=%llu)",
		                        int(m_), int(bits_), double(eta_), (unsigned long long)dim_);
	}
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	const idx_t book_floats = idx_t(m_) * k * sub_dim;
	const idx_t book_bytes = book_floats * sizeof(float);
	if (size != header + book_bytes) {
		throw InternalException("ScannQuantizer::Deserialize: blob size mismatch (got %llu, expected %llu)",
		                        (unsigned long long)size, (unsigned long long)(header + book_bytes));
	}
	codebook_.assign(book_floats, 0.0f);
	if (book_bytes > 0) {
		std::memcpy(codebook_.data(), in + header, book_bytes);
	}
	trained_ = true;
}

} // namespace scann
} // namespace vindex
} // namespace duckdb
