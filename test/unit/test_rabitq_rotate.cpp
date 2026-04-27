// Unit tests for RaBitQ's deterministic random rotation (src/quant/rabitq/
// rabitq_rotate.cpp). Three properties are load-bearing for the rest of the
// quantizer and have to be checked here:
//
//   1. R is orthogonal: R*Rᵀ ≈ I (and therefore ‖Rx‖ = ‖x‖).
//   2. Given the same (dim, seed), the matrix is bitwise identical across
//      constructions — otherwise persisted indexes break.
//   3. Serialize → Deserialize round-trips, producing a rotation that maps
//      vectors identically to the original.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "duckdb/common/vector.hpp"

#include "../../src/quant/rabitq/rabitq_rotate.hpp"

#include <cmath>
#include <random>

using Catch::Matchers::WithinAbs;
using duckdb::data_t;
using duckdb::idx_t;
using duckdb::vector;
using duckdb::vindex::rabitq::RandomRotation;

namespace {

// Apply R, then Rᵀ, and check we're back at x. Works because R is orthogonal.
void RequireRoundTrip(const RandomRotation &R, const float *x, float tol) {
	vector<float> tmp(R.Dim());
	vector<float> back(R.Dim());
	R.Apply(x, tmp.data());
	R.ApplyTranspose(tmp.data(), back.data());
	for (idx_t i = 0; i < R.Dim(); i++) {
		REQUIRE_THAT(back[i], WithinAbs(x[i], tol));
	}
}

} // namespace

TEST_CASE("RandomRotation preserves vector norm (orthogonality)", "[rabitq][rotate][unit]") {
	RandomRotation R(32, 0xC0FFEE);
	std::mt19937_64 rng(42);
	std::normal_distribution<float> nd(0.0f, 1.0f);

	for (int trial = 0; trial < 5; trial++) {
		vector<float> x(32);
		float norm_sq = 0.0f;
		for (idx_t i = 0; i < 32; i++) {
			x[i] = nd(rng);
			norm_sq += x[i] * x[i];
		}
		vector<float> y(32);
		R.Apply(x.data(), y.data());
		float rot_norm_sq = 0.0f;
		for (idx_t i = 0; i < 32; i++) {
			rot_norm_sq += y[i] * y[i];
		}
		REQUIRE_THAT(rot_norm_sq, WithinAbs(norm_sq, 1e-3f));
	}
}

TEST_CASE("RandomRotation Apply followed by ApplyTranspose is identity", "[rabitq][rotate][unit]") {
	RandomRotation R(16, 7);
	const float e0[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	const float e5[16] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	RequireRoundTrip(R, e0, 1e-4f);
	RequireRoundTrip(R, e5, 1e-4f);
}

TEST_CASE("RandomRotation rows are orthonormal (R Rᵀ ≈ I)", "[rabitq][rotate][unit]") {
	RandomRotation R(24, 123);
	vector<float> row_i(24), row_j_rot(24);
	for (idx_t i = 0; i < 24; i++) {
		// Feed unit vector e_i; the result is the i-th column of R (= the
		// i-th row of Rᵀ). Two columns are orthogonal iff Rᵀ has orthonormal
		// rows, iff R Rᵀ = I.
		vector<float> e(24, 0.0f);
		e[i] = 1.0f;
		R.Apply(e.data(), row_i.data());
		float norm_sq = 0.0f;
		for (idx_t k = 0; k < 24; k++) {
			norm_sq += row_i[k] * row_i[k];
		}
		REQUIRE_THAT(norm_sq, WithinAbs(1.0f, 1e-4f));

		for (idx_t j = i + 1; j < 24; j++) {
			vector<float> ej(24, 0.0f);
			ej[j] = 1.0f;
			R.Apply(ej.data(), row_j_rot.data());
			float dot = 0.0f;
			for (idx_t k = 0; k < 24; k++) {
				dot += row_i[k] * row_j_rot[k];
			}
			REQUIRE_THAT(dot, WithinAbs(0.0f, 1e-4f));
		}
	}
}

TEST_CASE("RandomRotation is deterministic for a fixed (dim, seed)", "[rabitq][rotate][unit]") {
	RandomRotation a(40, 0xBADC0DE);
	RandomRotation b(40, 0xBADC0DE);
	vector<float> x(40);
	for (idx_t i = 0; i < 40; i++) {
		x[i] = static_cast<float>(i) * 0.25f - 4.0f;
	}
	vector<float> ya(40), yb(40);
	a.Apply(x.data(), ya.data());
	b.Apply(x.data(), yb.data());
	for (idx_t i = 0; i < 40; i++) {
		REQUIRE(ya[i] == yb[i]);
	}
}

TEST_CASE("RandomRotation differs for different seeds", "[rabitq][rotate][unit]") {
	RandomRotation a(16, 1);
	RandomRotation b(16, 2);
	const float x[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	vector<float> ya(16), yb(16);
	a.Apply(x, ya.data());
	b.Apply(x, yb.data());

	// There should be at least one coordinate where the two rotations differ
	// by more than float32 rounding noise.
	bool any_diff = false;
	for (idx_t i = 0; i < 16; i++) {
		if (std::fabs(ya[i] - yb[i]) > 1e-4f) {
			any_diff = true;
			break;
		}
	}
	REQUIRE(any_diff);
}

TEST_CASE("RandomRotation Serialize round-trips and reproduces matrix",
          "[rabitq][rotate][unit]") {
	RandomRotation src(32, 0xFEEDFACE);
	vector<data_t> blob;
	src.Serialize(blob);
	REQUIRE(blob.size() == sizeof(uint64_t) * 2);

	RandomRotation dst;
	dst.Deserialize(blob.data(), blob.size());
	REQUIRE(dst.Dim() == 32);
	REQUIRE(dst.Seed() == 0xFEEDFACE);

	const float x[32] = {-1.5f, 0.5f, 2.0f, -0.25f, 1.0f, 0.0f, 3.0f, 4.0f,
	                     -2.5f, 1.25f, 0.75f, -0.5f, 2.0f, 1.0f, -1.0f, 0.5f,
	                     0.0f, 1.5f, -2.0f, 0.25f, -0.75f, 1.75f, 3.25f, -1.25f,
	                     2.5f, -0.5f, 0.0f, 1.0f, -1.0f, 2.0f, -3.0f, 1.5f};
	vector<float> y_src(32), y_dst(32);
	src.Apply(x, y_src.data());
	dst.Apply(x, y_dst.data());
	for (idx_t i = 0; i < 32; i++) {
		REQUIRE(y_src[i] == y_dst[i]);
	}
}
