#include "rabitq_kernels.hpp"

#include "duckdb/common/exception.hpp"

#include <cstdint>

namespace duckdb {
namespace vindex {
namespace rabitq {
namespace kernels {

// ---------------------------------------------------------------------------
// 1-bit: Σ (2·bit[i] - 1) · q[i] = 2·Σ q[i where bit=1] - Σ q[i]
//
// The identity above halves the work vs the naive "branch per bit" loop:
// we precompute the full query sum once, then accumulate only the set-bit
// contributions. Both parts autovectorize well on NEON and AVX2 because the
// inner loop is a pure float add over a compile-time-bounded stripe of 8
// values per byte.
// ---------------------------------------------------------------------------
float Dot1Bit(const_data_ptr_t packed, const float *query, idx_t dim) {
	float sum_q = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		sum_q += query[i];
	}

	float set_sum = 0.0f;
	const idx_t full_bytes = dim / 8;
	const idx_t rem_bits = dim % 8;
	idx_t dim_off = 0;
	for (idx_t b = 0; b < full_bytes; b++) {
		const uint8_t byte = packed[b];
		const float *q = query + dim_off;
		// Fully unrolled — the compiler vectorizes this into a masked add on
		// every platform we target.
		if (byte & 0x01) set_sum += q[0];
		if (byte & 0x02) set_sum += q[1];
		if (byte & 0x04) set_sum += q[2];
		if (byte & 0x08) set_sum += q[3];
		if (byte & 0x10) set_sum += q[4];
		if (byte & 0x20) set_sum += q[5];
		if (byte & 0x40) set_sum += q[6];
		if (byte & 0x80) set_sum += q[7];
		dim_off += 8;
	}
	if (rem_bits > 0) {
		const uint8_t byte = packed[full_bytes];
		for (idx_t r = 0; r < rem_bits; r++) {
			if ((byte >> r) & 1u) {
				set_sum += query[dim_off + r];
			}
		}
	}
	return 2.0f * set_sum - sum_q;
}

// ---------------------------------------------------------------------------
// 8-bit: codes stored as signed int8_t, one byte per dim. A plain fused
// multiply-add loop; modern compilers emit NEON / AVX2 widening-multiply
// here without further help.
// ---------------------------------------------------------------------------
float DotI8(const_data_ptr_t packed, const float *query, idx_t dim) {
	const int8_t *codes = reinterpret_cast<const int8_t *>(packed);
	float acc = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		acc += static_cast<float>(codes[i]) * query[i];
	}
	return acc;
}

// ---------------------------------------------------------------------------
// bits ∈ {2,3,4,5,7}: scalar unpack. Each code straddles at most two bytes
// (bits ≤ 8), and we sign-extend by masking the `bits`-wide value and OR-ing
// the one-bit above the MSB with a pre-computed extension mask.
//
// Kept as one parametric loop rather than seven specializations because these
// paths are rarely the query-throughput hot spot (bits=3 carries production
// traffic via bits=1 + rerank, and bits=8 has its own kernel above).
// ---------------------------------------------------------------------------
float DotPacked(const_data_ptr_t packed, const float *query, idx_t dim, uint8_t bits) {
	const uint32_t mask = (1u << bits) - 1u;
	const uint32_t sign_bit = 1u << (bits - 1);
	const uint32_t ext = ~mask;

	float acc = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		const idx_t bit_off = i * bits;
		const idx_t byte_off = bit_off / 8;
		const uint8_t shift = static_cast<uint8_t>(bit_off % 8);
		uint32_t raw = static_cast<uint32_t>(packed[byte_off]) >> shift;
		const uint8_t used = static_cast<uint8_t>(8 - shift);
		if (used < bits) {
			raw |= static_cast<uint32_t>(packed[byte_off + 1]) << used;
		}
		raw &= mask;
		if (raw & sign_bit) {
			raw |= ext;
		}
		acc += static_cast<float>(static_cast<int32_t>(raw)) * query[i];
	}
	return acc;
}

float Dot(const_data_ptr_t packed, const float *query, idx_t dim, uint8_t bits) {
	switch (bits) {
	case 1:
		return Dot1Bit(packed, query, dim);
	case 8:
		return DotI8(packed, query, dim);
	case 2:
	case 3:
	case 4:
	case 5:
	case 7:
		return DotPacked(packed, query, dim, bits);
	default:
		throw InternalException("RabitqKernel::Dot: unsupported bits=%d", int(bits));
	}
}

} // namespace kernels
} // namespace rabitq
} // namespace vindex
} // namespace duckdb
