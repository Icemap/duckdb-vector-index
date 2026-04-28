#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/index_block_store.hpp"
#include "vindex/quantizer.hpp"

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// IvfCore — inverted-file ANN index.
//
// Training: k-means++ (Lloyd's) over a sample of float32 vectors produces
// `nlist` centroids. Each vector is then assigned to its nearest centroid
// and the (row_id, quantized_code) pair is appended to that centroid's
// in-memory posting list.
//
// Search: compute exact L2 from query to every centroid, keep the
// `nprobe` closest; then scan those posting lists, scoring each code with
// the quantizer's `EstimateDistance`. Returns `limit` candidate row_ids —
// exact rerank is handled by the upstream TopN operator.
//
// Storage: posting lists live in memory as byte arrays during normal
// operation. On SerializeState we write centroids + the concatenated
// posting-list bytes into a single caller-provided blob; the enclosing
// IvfIndex then hands that blob to its state stream via
// IndexBlockStore::BeginStream. This keeps us within the existing
// streaming API (which is overwrite-only, not append) without rewriting
// 1024 streams on every Insert.
// ---------------------------------------------------------------------------

struct IvfCoreParams {
	idx_t nlist = 1024;
	idx_t nprobe = 16;
	idx_t dim = 0;
	uint64_t seed = 42;
	// Lloyd iterations. 20 is plenty — k-means converges fast on typical
	// embedding distributions.
	idx_t kmeans_iters = 20;
};

class IvfCore {
public:
	IvfCore(IvfCoreParams params, Quantizer &quantizer, IndexBlockStore &store);
	~IvfCore();

	IvfCore(const IvfCore &) = delete;
	IvfCore &operator=(const IvfCore &) = delete;

	void Train(const float *samples, idx_t n);
	bool IsTrained() const {
		return trained_;
	}

	idx_t Insert(int64_t row_id, const float *vec);

	struct Candidate {
		float dist;
		int64_t row_id;
	};

	vector<Candidate> Search(const float *query_exact, const float *query_preproc, idx_t limit,
	                         idx_t nprobe_override = 0) const;

	idx_t Size() const {
		return inserted_;
	}

	idx_t NumCentroids() const {
		return params_.nlist;
	}

	const IvfCoreParams &Params() const {
		return params_;
	}

	// State (centroids + trained flag + posting bytes) serialized / restored
	// via the IvfIndex state stream.
	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

private:
	// Entry layout in each posting list:
	//   int64_t row_id
	//   uint8_t code[quantizer_.CodeSize()]
	idx_t EntrySize() const {
		return sizeof(int64_t) + code_size_;
	}

	idx_t AssignCentroid(const float *vec) const;

	IvfCoreParams params_;
	Quantizer &quantizer_;
	IndexBlockStore &store_;
	idx_t code_size_;

	bool trained_ = false;
	// nlist × dim float32 centroids (row-major).
	vector<float> centroids_;
	// Per-list in-memory entry buffer. postings_[c] holds
	// postings_count_[c] × EntrySize() bytes (row_id little-endian + code).
	vector<vector<data_t>> postings_;
	vector<uint64_t> postings_count_;
	idx_t inserted_ = 0;
};

} // namespace vindex
} // namespace duckdb
