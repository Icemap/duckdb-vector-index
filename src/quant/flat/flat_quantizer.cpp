#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/string_util.hpp"

#include "vindex/pq_quantizer.hpp"
#include "vindex/quantizer.hpp"

#include "../rabitq/rabitq_quantizer.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <cmath>
#include <cstring>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// FlatQuantizer — identity quantizer. `code` is just the raw float32 vector,
// so EstimateDistance is exact. Useful as (a) the default when no quantizer
// is requested and (b) a ground-truth baseline for RaBitQ recall regressions.
// ---------------------------------------------------------------------------

namespace {

class FlatQuantizer : public Quantizer {
public:
	FlatQuantizer(MetricKind metric, idx_t dim) : metric_(metric), dim_(dim) {
	}

	void Train(const float *, idx_t, idx_t) override {
	}

	void Encode(const float *vec, data_ptr_t code_out) const override {
		std::memcpy(code_out, vec, dim_ * sizeof(float));
	}

	float EstimateDistance(const_data_ptr_t code, const float *query_preproc) const override {
		auto stored = reinterpret_cast<const simsimd_f32_t *>(code);
		auto q = reinterpret_cast<const simsimd_f32_t *>(query_preproc);
		simsimd_distance_t d = 0;
		switch (metric_) {
		case MetricKind::L2SQ:
			simsimd_l2sq_f32(stored, q, dim_, &d);
			return float(d);
		case MetricKind::COSINE: {
			// PreprocessQuery already normalized the query, so we only need
			// |stored| here. We avoid simsimd_cos_f32 because it uses an
			// approximate rsqrt with ~1e-3 error; two dots + a scalar sqrt
			// stays within 1e-5 and is the same number of SIMD passes.
			simsimd_distance_t dot = 0;
			simsimd_distance_t norm_sq = 0;
			simsimd_dot_f32(stored, q, dim_, &dot);
			simsimd_dot_f32(stored, stored, dim_, &norm_sq);
			if (norm_sq == 0) {
				return 1.0f;
			}
			return 1.0f - float(dot) / std::sqrt(float(norm_sq));
		}
		case MetricKind::IP:
			// DuckDB's array_negative_inner_product: `-sum(a*b)`. simsimd_dot
			// returns the positive dot, so negate.
			simsimd_dot_f32(stored, q, dim_, &d);
			return -float(d);
		}
		throw InternalException("FlatQuantizer: unreachable MetricKind");
	}

	float CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const override {
		auto a = reinterpret_cast<const simsimd_f32_t *>(code_a);
		auto b = reinterpret_cast<const simsimd_f32_t *>(code_b);
		simsimd_distance_t d = 0;
		switch (metric_) {
		case MetricKind::L2SQ:
			simsimd_l2sq_f32(a, b, dim_, &d);
			return float(d);
		case MetricKind::COSINE: {
			simsimd_distance_t dot = 0, na = 0, nb = 0;
			simsimd_dot_f32(a, b, dim_, &dot);
			simsimd_dot_f32(a, a, dim_, &na);
			simsimd_dot_f32(b, b, dim_, &nb);
			const float denom = std::sqrt(float(na)) * std::sqrt(float(nb));
			if (denom == 0.0f) {
				return 1.0f;
			}
			return 1.0f - float(dot) / denom;
		}
		case MetricKind::IP:
			simsimd_dot_f32(a, b, dim_, &d);
			return -float(d);
		}
		throw InternalException("FlatQuantizer: unreachable MetricKind");
	}

	void PreprocessQuery(const float *query, float *out) const override {
		if (metric_ == MetricKind::COSINE) {
			// Pre-normalize once so per-code EstimateDistance only pays the
			// stored-vector norm. Use simsimd to compute |q|^2.
			simsimd_distance_t norm_sq = 0;
			simsimd_dot_f32(reinterpret_cast<const simsimd_f32_t *>(query),
			                reinterpret_cast<const simsimd_f32_t *>(query), dim_, &norm_sq);
			const float inv = norm_sq == 0.0 ? 0.0f : 1.0f / std::sqrt(float(norm_sq));
			for (idx_t i = 0; i < dim_; i++) {
				out[i] = query[i] * inv;
			}
			return;
		}
		std::memcpy(out, query, dim_ * sizeof(float));
	}

	idx_t CodeSize() const override {
		return dim_ * sizeof(float);
	}
	idx_t QueryWorkspaceSize() const override {
		return dim_;
	}
	MetricKind Metric() const override {
		return metric_;
	}
	QuantizerKind Kind() const override {
		return QuantizerKind::FLAT;
	}

	void Serialize(vector<data_t> &out) const override {
		// {kind:u8, metric:u8, dim:u64}
		out.resize(sizeof(uint8_t) * 2 + sizeof(uint64_t));
		auto ptr = out.data();
		ptr[0] = static_cast<uint8_t>(QuantizerKind::FLAT);
		ptr[1] = static_cast<uint8_t>(metric_);
		const uint64_t d = dim_;
		std::memcpy(ptr + 2, &d, sizeof(d));
	}

	void Deserialize(const_data_ptr_t in, idx_t size) override {
		if (size < sizeof(uint8_t) * 2 + sizeof(uint64_t)) {
			throw InternalException("FlatQuantizer::Deserialize: blob too small (%llu)", (unsigned long long)size);
		}
		const auto kind = static_cast<QuantizerKind>(in[0]);
		if (kind != QuantizerKind::FLAT) {
			throw InternalException("FlatQuantizer::Deserialize: wrong kind tag %d", int(in[0]));
		}
		metric_ = static_cast<MetricKind>(in[1]);
		uint64_t d;
		std::memcpy(&d, in + 2, sizeof(d));
		dim_ = d;
	}

private:
	MetricKind metric_;
	idx_t dim_;
};

} // namespace

// ---------------------------------------------------------------------------
// Factory — parses `WITH (quantizer = '...')` option.
// Dispatches to the right concrete Quantizer. Ships FLAT / RABITQ / PQ.
// ---------------------------------------------------------------------------
unique_ptr<Quantizer> CreateQuantizer(const case_insensitive_map_t<Value> &options, MetricKind metric, idx_t dim) {
	string name = "flat";
	auto it = options.find("quantizer");
	if (it != options.end()) {
		if (it->second.type() != LogicalType::VARCHAR) {
			throw BinderException("vindex: 'quantizer' option must be a string");
		}
		name = StringUtil::Lower(it->second.GetValue<string>());
	}

	if (name == "flat") {
		return make_uniq<FlatQuantizer>(metric, dim);
	}
	if (name == "rabitq") {
		// Default 3 bits: the recall × memory sweet spot per README §"Quantizer
		// bits vs recall". Supported widths: {1,2,3,4,5,7,8} (6 is pareto-dominated).
		uint8_t bits = 3;
		auto bit_it = options.find("bits");
		if (bit_it != options.end()) {
			if (bit_it->second.type() != LogicalType::INTEGER) {
				throw BinderException("vindex: rabitq 'bits' option must be an integer");
			}
			const int32_t b = bit_it->second.GetValue<int32_t>();
			switch (b) {
			case 1:
			case 2:
			case 3:
			case 4:
			case 5:
			case 7:
			case 8:
				bits = static_cast<uint8_t>(b);
				break;
			default:
				throw BinderException("vindex: rabitq bits=%d is not supported (expected one of 1,2,3,4,5,7,8)", b);
			}
		}
		// bits < 3 is a coarse filter that is only useful with rerank. Emit a
		// warning here so users who pick bits=1/2 without wiring rerank see
		// the recall penalty up-front instead of blaming the extension later.
		// See README.md §"Quantizer bits vs recall" for the numbers.
		if (bits < 3) {
			Printer::PrintF(OutputStream::STREAM_STDERR,
			                "vindex WARNING: rabitq bits=%d has ~40%% Recall@10 without rerank. "
			                "Pair with a larger rerank multiple (>=20) or use bits=3 (default) for "
			                ">=0.99 Recall@10. See README.md §'Quantizer bits vs recall'.\n",
			                int(bits));
		}
		return make_uniq<rabitq::RabitqQuantizer>(metric, dim, bits);
	}
	if (name == "pq") {
		// Defaults: bits=8 (256 centroids/slot, <1% distance error on SIFT-
		// scale features), m = max(1, dim / 4) — i.e. each sub-vector is 4
		// floats, a common sweet spot for the k-means cost vs granularity
		// trade-off. Users who need smaller codes bump m; who need lower
		// training cost drop it.
		uint8_t bits = 8;
		auto bit_it = options.find("bits");
		if (bit_it != options.end()) {
			if (bit_it->second.type() != LogicalType::INTEGER) {
				throw BinderException("vindex: pq 'bits' option must be an integer");
			}
			const int32_t b = bit_it->second.GetValue<int32_t>();
			if (b != 4 && b != 8) {
				throw BinderException("vindex: pq bits=%d is not supported (expected 4 or 8)", b);
			}
			bits = static_cast<uint8_t>(b);
		}

		idx_t m = dim >= 4 ? dim / 4 : 1;
		auto m_it = options.find("m");
		if (m_it != options.end()) {
			if (m_it->second.type() != LogicalType::INTEGER) {
				throw BinderException("vindex: pq 'm' option must be an integer");
			}
			const int32_t mm = m_it->second.GetValue<int32_t>();
			if (mm <= 0 || mm > 255) {
				throw BinderException("vindex: pq m=%d out of range (1..255)", mm);
			}
			m = static_cast<idx_t>(mm);
		}
		if (dim % m != 0) {
			throw BinderException("vindex: pq requires dim (%llu) to be divisible by m (%llu)",
			                      (unsigned long long)dim, (unsigned long long)m);
		}
		return make_uniq<pq::PqQuantizer>(metric, dim, static_cast<uint8_t>(m), bits);
	}
	throw BinderException("vindex: unknown quantizer '%s' (expected 'flat', 'rabitq', or 'pq')", name);
}

} // namespace vindex
} // namespace duckdb
