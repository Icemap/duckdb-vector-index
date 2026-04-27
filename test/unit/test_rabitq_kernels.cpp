// Unit tests for src/quant/rabitq/rabitq_kernels.cpp.
//
// The kernels are the only place in the RaBitQ path where we sacrifice
// readability for SIMD-friendly inner loops. The tests here pin each kernel
// against an obvious scalar reference, fixed inputs + random inputs, so a
// future SIMD rewrite (e.g. NEON / AVX2 intrinsics) can't regress the math.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector.hpp"

#include "../../src/quant/rabitq/rabitq_kernels.hpp"

#include <cstdint>
#include <cstring>
#include <random>

using Catch::Matchers::WithinAbs;
using duckdb::data_t;
using duckdb::idx_t;
using duckdb::vector;

namespace {

// Reference scalar: Σ (2·bit[i] - 1) · q[i].
float Ref1Bit(const uint8_t *packed, const float *q, idx_t dim) {
	float acc = 0.0f;
	for (idx_t i = 0; i < dim; i++) {
		const uint8_t bit = (packed[i / 8] >> (i % 8)) & 1u;
		acc += (bit ? 1.0f : -1.0f) * q[i];
	}
	return acc;
}

// Reference scalar: Σ code[i] · q[i] for signed b-bit codes, LSB-first.
float RefPacked(const uint8_t *packed, const float *q, idx_t dim, uint8_t bits) {
	const uint32_t mask = (1u << bits) - 1u;
	const uint32_t sign = 1u << (bits - 1);
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
		int32_t code = (raw & sign) ? static_cast<int32_t>(raw | ~mask) : static_cast<int32_t>(raw);
		acc += static_cast<float>(code) * q[i];
	}
	return acc;
}

void PackSigned(const int32_t *codes, idx_t dim, uint8_t bits, uint8_t *out) {
	const idx_t byte_count = (dim * bits + 7) / 8;
	std::memset(out, 0, byte_count);
	const uint32_t mask = (1u << bits) - 1u;
	for (idx_t i = 0; i < dim; i++) {
		const uint32_t code = static_cast<uint32_t>(codes[i]) & mask;
		const idx_t bit_off = i * bits;
		const idx_t byte_off = bit_off / 8;
		const uint8_t shift = static_cast<uint8_t>(bit_off % 8);
		out[byte_off] |= static_cast<uint8_t>((code << shift) & 0xFFu);
		const uint8_t used = static_cast<uint8_t>(8 - shift);
		if (used < bits) {
			out[byte_off + 1] |= static_cast<uint8_t>((code >> used) & 0xFFu);
		}
	}
}

} // namespace

TEST_CASE("Dot1Bit matches scalar reference on fixed inputs", "[rabitq][kernels][unit]") {
	// byte0 = 0b1001'1011 → LSB-first dims 0..7 = 1,1,0,1,1,0,0,1
	// byte1 = 0b00000011 → LSB-first dims 8..9 = 1,1
	// signs:                +1 +1 -1 +1 +1 -1 -1 +1 +1 +1
	const idx_t dim = 10;
	uint8_t packed[2] = {0b10011011, 0b11};
	const float q[dim] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	// +1 +2 -3 +4 +5 -6 -7 +8 +9 +10 = 23
	const float expected = 23.0f;
	REQUIRE_THAT(duckdb::vindex::rabitq::kernels::Dot1Bit(packed, q, dim), WithinAbs(expected, 1e-5f));
	REQUIRE_THAT(Ref1Bit(packed, q, dim), WithinAbs(expected, 1e-5f));
}

TEST_CASE("Dot1Bit handles dim=0 and dim below one byte", "[rabitq][kernels][unit]") {
	uint8_t packed[1] = {0b101}; // bits 0,1,2 = 1,0,1 → signs +,-,+
	const float q[3] = {10, 20, 30};
	REQUIRE_THAT(duckdb::vindex::rabitq::kernels::Dot1Bit(packed, q, 3), WithinAbs(10.0f - 20.0f + 30.0f, 1e-5f));
	REQUIRE(duckdb::vindex::rabitq::kernels::Dot1Bit(packed, q, 0) == 0.0f);
}

TEST_CASE("Dot1Bit random matches scalar reference across a spread of dims",
          "[rabitq][kernels][unit]") {
	std::mt19937_64 rng(0xBADBEEF);
	std::uniform_int_distribution<int> bit(0, 1);
	std::normal_distribution<float> nd(0.0f, 1.0f);

	for (idx_t dim : {1, 7, 8, 9, 16, 17, 63, 64, 128, 129}) {
		const idx_t bytes = (dim + 7) / 8;
		vector<uint8_t> packed(bytes, 0);
		for (idx_t i = 0; i < dim; i++) {
			if (bit(rng)) {
				packed[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
			}
		}
		vector<float> q(dim);
		for (idx_t i = 0; i < dim; i++) {
			q[i] = nd(rng);
		}
		const float got = duckdb::vindex::rabitq::kernels::Dot1Bit(packed.data(), q.data(), dim);
		const float want = Ref1Bit(packed.data(), q.data(), dim);
		INFO("dim=" << dim);
		REQUIRE_THAT(got, WithinAbs(want, 1e-4f));
	}
}

TEST_CASE("DotI8 matches scalar reference on fixed inputs", "[rabitq][kernels][unit]") {
	const int8_t codes[] = {1, -2, 3, -4};
	const float q[] = {10, 20, 30, 40};
	// 1·10 + (-2)·20 + 3·30 + (-4)·40 = 10 - 40 + 90 - 160 = -100
	REQUIRE_THAT(duckdb::vindex::rabitq::kernels::DotI8(reinterpret_cast<const data_t *>(codes), q, 4),
	             WithinAbs(-100.0f, 1e-5f));
}

TEST_CASE("DotPacked matches scalar reference for every supported bit width",
          "[rabitq][kernels][unit]") {
	std::mt19937_64 rng(42);
	std::normal_distribution<float> nd(0.0f, 1.0f);

	for (uint8_t bits : {2, 3, 4, 5, 7}) {
		const int32_t lo = -(1 << (bits - 1));
		const int32_t hi = (1 << (bits - 1)) - 1;
		std::uniform_int_distribution<int32_t> dist(lo, hi);

		for (idx_t dim : {1, 7, 8, 9, 16, 64, 128}) {
			vector<int32_t> codes(dim);
			for (idx_t i = 0; i < dim; i++) {
				codes[i] = dist(rng);
			}
			vector<uint8_t> packed((dim * bits + 7) / 8, 0);
			PackSigned(codes.data(), dim, bits, packed.data());

			vector<float> q(dim);
			for (idx_t i = 0; i < dim; i++) {
				q[i] = nd(rng);
			}

			const float got =
			    duckdb::vindex::rabitq::kernels::DotPacked(packed.data(), q.data(), dim, bits);
			const float want = RefPacked(packed.data(), q.data(), dim, bits);
			INFO("bits=" << int(bits) << " dim=" << dim);
			REQUIRE_THAT(got, WithinAbs(want, 1e-3f));

			// Also cross-check the naive formula using the raw int codes.
			float naive = 0.0f;
			for (idx_t i = 0; i < dim; i++) {
				naive += static_cast<float>(codes[i]) * q[i];
			}
			REQUIRE_THAT(got, WithinAbs(naive, 1e-3f));
		}
	}
}

TEST_CASE("Dot dispatch routes to the right kernel", "[rabitq][kernels][unit]") {
	using duckdb::vindex::rabitq::kernels::Dot;
	// bits=1
	{
		const uint8_t packed[1] = {0b101};
		const float q[3] = {1, 1, 1};
		REQUIRE_THAT(Dot(packed, q, 3, 1), WithinAbs(1.0f - 1.0f + 1.0f, 1e-5f));
	}
	// bits=3: codes = {3, -4, 1} (each fits in 3 bits signed).
	{
		const int32_t c[] = {3, -4, 1};
		uint8_t packed[2] = {0};
		PackSigned(c, 3, 3, packed);
		const float q[3] = {1, 1, 1};
		REQUIRE_THAT(Dot(packed, q, 3, 3), WithinAbs(0.0f, 1e-5f));
	}
	// bits=8
	{
		const int8_t c[] = {5, -6, 7, -8};
		const float q[4] = {1, 1, 1, 1};
		REQUIRE_THAT(Dot(reinterpret_cast<const uint8_t *>(c), q, 4, 8),
		             WithinAbs(-2.0f, 1e-5f));
	}
	// Unsupported bits → InternalException.
	{
		const uint8_t packed[1] = {0};
		const float q[1] = {0};
		REQUIRE_THROWS_AS(Dot(packed, q, 1, 6), duckdb::InternalException);
	}
}
