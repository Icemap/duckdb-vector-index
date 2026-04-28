#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/types/value.hpp"

#include "vindex/metric.hpp"

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// Quantizer — algorithm-agnostic compressor for FLOAT[d] vectors.
//
// Any ANN algorithm (HNSW, IVF, DiskANN, ...) can accept a Quantizer; it turns
// full-precision vectors into byte codes used for fast approximate distance,
// with the authoritative vector re-read from the original ARRAY column at
// rerank time. This keeps our storage footprint low without duplicating data.
//
// Concrete implementations:
//   - FlatQuantizer   (identity, stores float32)
//   - PqQuantizer     (classical product quantization)          [M2+]
//   - RabitqQuantizer (RaBitQ / ExRaBitQ, default at bits=3)    [M1]
// ---------------------------------------------------------------------------

enum class QuantizerKind : uint8_t {
	FLAT = 0,
	PQ = 1,
	RABITQ = 2,
	SCANN = 3,
};

class Quantizer {
public:
	virtual ~Quantizer() = default;

	// One-shot training on a sample of vectors. Expected to be called once at
	// CREATE INDEX time, before any Encode(). Implementations that don't need
	// training (Flat) return immediately.
	virtual void Train(const float *samples, idx_t n, idx_t dim) = 0;

	// Encode a single vector into a byte buffer of at least CodeSize() bytes.
	virtual void Encode(const float *vec, data_ptr_t code_out) const = 0;

	// Estimated distance between a query (already preprocessed via
	// PreprocessQuery) and a stored code. Metric is fixed at construction.
	virtual float EstimateDistance(const_data_ptr_t code, const float *query_preproc) const = 0;

	// Estimated distance between two stored codes. Required by graph-building
	// heuristics that need to decide whether to keep an edge based on
	// pairwise neighbor distances (HNSW Algorithm 4). Implementations that
	// cannot provide a fast code-to-code distance (e.g. RaBitQ at narrow
	// bit-widths where codes lose too much information) may decode + rescore
	// — callers treat this as insert-time-only and expect it to be slower
	// than EstimateDistance.
	virtual float CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const = 0;

	// Prepare a query vector for repeated use (e.g. apply the same random
	// rotation RaBitQ uses at encode time). `out` must have room for at least
	// QueryWorkspaceSize() floats.
	virtual void PreprocessQuery(const float *query, float *out) const = 0;

	virtual idx_t CodeSize() const = 0;         // bytes per vector
	virtual idx_t QueryWorkspaceSize() const = 0; // floats
	virtual MetricKind Metric() const = 0;
	virtual QuantizerKind Kind() const = 0;

	// Serialize learned parameters (rotation matrix, PQ centroids, ...) to a
	// self-describing blob. The blob is appended to the index's block store
	// as part of SerializeToDisk/SerializeToWAL.
	virtual void Serialize(vector<data_t> &out) const = 0;
	virtual void Deserialize(const_data_ptr_t in, idx_t size) = 0;
};

// Factory. Parses `WITH (quantizer = '...', bits = N, ...)` options.
unique_ptr<Quantizer> CreateQuantizer(const case_insensitive_map_t<Value> &options, MetricKind metric, idx_t dim);

} // namespace vindex
} // namespace duckdb
