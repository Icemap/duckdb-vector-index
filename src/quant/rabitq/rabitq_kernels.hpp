#pragma once

#include "duckdb/common/typedefs.hpp"

namespace duckdb {
namespace vindex {
namespace rabitq {
namespace kernels {

// ---------------------------------------------------------------------------
// RaBitQ inner-product kernels. These compute the "numerator" term
//   dot = Σ unpacked_code[i] * query_rot[i]
// for every supported bit width. The caller multiplies by the per-vector
// scale (1/√d · α / β for bits=1, Δ for bits≥2) and folds it into the
// L2SQ estimator.
//
// Each kernel takes the packed code starting at `packed` (bit 0 of dim 0
// is the LSB of byte 0; codes are LSB-first two's-complement for bits≥2).
// `query` points to the rotated query workspace produced by
// RabitqQuantizer::PreprocessQuery (first `dim` floats).
//
// The kernels are split into one function per bit width so each has a tight
// hot loop the compiler can autovectorize; the 1-bit and 8-bit paths are the
// ones that actually carry production traffic (coarse filter and full
// fidelity respectively) and get manual attention first.
// ---------------------------------------------------------------------------

// bits == 1: sign-packed. Returns Σ (2·bit[i] - 1) · query[i].
//   – Does NOT include the 1/√d normalization; caller applies it.
float Dot1Bit(const_data_ptr_t packed, const float *query, idx_t dim);

// bits == 8: codes stored as int8_t, one byte per dim.
float DotI8(const_data_ptr_t packed, const float *query, idx_t dim);

// bits ∈ {2,3,4,5,7}: scalar unpack. Slower but the code densities where the
// recall-memory trade-off favours bits=3/4 over bits=1-plus-rerank are not
// typically the query-throughput hot spot.
float DotPacked(const_data_ptr_t packed, const float *query, idx_t dim, uint8_t bits);

// Dispatch: pick the right kernel for a given `bits`.
float Dot(const_data_ptr_t packed, const float *query, idx_t dim, uint8_t bits);

} // namespace kernels
} // namespace rabitq
} // namespace vindex
} // namespace duckdb
