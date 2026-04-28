#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/index_block_store.hpp"
#include "vindex/quantizer.hpp"

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// SpannCore — inverted-file index with replica (closure) writes.
//
// Layout-wise this is IVF: k-means++ centroids and per-centroid posting lists
// of (row_id, code) entries. The SPANN-specific innovations (Chen et al.,
// NeurIPS 2021) that this module implements are:
//
//   1. **Replica / closure assignment**. At insert time the vector is written
//      into the top `replica_count` nearest centroids *conditional* on their
//      L2 distance falling within `closure_factor × d_best`. That way a
//      point near the boundary of several Voronoi cells ends up indexed in
//      all of them, which restores the recall that a strictly-partitioned
//      IVF would lose.
//
//   2. **Query-time deduplication**. Because a row can appear in many probed
//      posting lists, the top-k heap dedupes by row_id before returning.
//
// SPANN's paper also specifies disk-backed posting lists with a posting
// budget; that is a storage/throughput optimisation on top of the same
// algorithm, deferred in this module (postings live in RAM like IvfCore).
// The `IndexBlockStore` reference is taken to keep the constructor
// signature stable for when we do plumb per-list blocks through it.
// ---------------------------------------------------------------------------

struct SpannCoreParams {
	idx_t nlist = 1024;
	idx_t nprobe = 16;
	idx_t dim = 0;
	uint64_t seed = 42;
	idx_t kmeans_iters = 20;

	// SPANN replica parameters. Defaults from the paper §3.1: 8 replicas
	// with a 1.1× closure factor. A closure factor of 1.0 collapses to
	// strict IVF; higher values trade index size for recall.
	idx_t replica_count = 8;
	float closure_factor = 1.1f;
};

class SpannCore {
public:
	SpannCore(SpannCoreParams params, Quantizer &quantizer, IndexBlockStore &store);
	~SpannCore();

	SpannCore(const SpannCore &) = delete;
	SpannCore &operator=(const SpannCore &) = delete;

	void Train(const float *samples, idx_t n);
	bool IsTrained() const {
		return trained_;
	}

	// Returns the primary (closest) centroid id. Replicas are written
	// transparently, but the caller uses the primary to record a row → cell
	// map for Delete bookkeeping.
	idx_t Insert(int64_t row_id, const float *vec);

	struct Candidate {
		float dist;
		int64_t row_id;
	};

	vector<Candidate> Search(const float *query_exact, const float *query_preproc, idx_t limit,
	                         idx_t nprobe_override = 0) const;

	// Count of *unique rows* inserted (replicas do not inflate this).
	idx_t Size() const {
		return inserted_;
	}

	idx_t NumCentroids() const {
		return params_.nlist;
	}

	// Total posting-list entries across every list. Always ≥ Size(); the
	// ratio Entries() / Size() is the average replica count actually written.
	idx_t Entries() const;

	const SpannCoreParams &Params() const {
		return params_;
	}

	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

private:
	// Entry layout: int64_t row_id followed by the quantizer's encoded code.
	idx_t EntrySize() const {
		return sizeof(int64_t) + code_size_;
	}

	// Returns the centroid ids to write this vector into. First element is the
	// closest centroid; subsequent entries are "closure" centroids whose L2
	// distance is ≤ `closure_factor × d_best`. Capped at `replica_count`.
	vector<idx_t> ChooseCentroids(const float *vec) const;

	SpannCoreParams params_;
	Quantizer &quantizer_;
	IndexBlockStore &store_;
	idx_t code_size_;

	bool trained_ = false;
	vector<float> centroids_;
	vector<vector<data_t>> postings_;
	vector<uint64_t> postings_count_;
	idx_t inserted_ = 0;
};

} // namespace vindex
} // namespace duckdb
