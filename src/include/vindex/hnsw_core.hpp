#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/index_block_store.hpp"
#include "vindex/quantizer.hpp"

#include <cstdint>
#include <random>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// HnswCore — DuckDB-independent HNSW implementation.
//
// Why not usearch? The reference `HNSWIndex` wraps `unum::usearch::index_dense`
// and dumps it opaquely into a LinkedBlock chain (see ref/duckdb-vss/
// src/hnsw/hnsw_index.cpp:45-144). That design made sense at M0 — it was a
// port — but three things force a rewrite for M1+:
//
//   1. Quantization. usearch's scalar types are (f32, f16, i8, b1). RaBitQ's
//      b-bit packed codes and the upcoming PQ codebooks don't fit any of those
//      templates, and its metric API takes a `scalar_kind_t` enum that we'd
//      have to fork.
//   2. Rerank / fine-search. For RaBitQ we need to re-score the top-ef via the
//      authoritative float vector. usearch hides the candidate list behind its
//      iterator so there's no extension point.
//   3. Block storage. DiskANN and SPANN (M3/M4) need per-node block addressing
//      — IndexBlockStore is the substrate they share with HNSW. The usearch
//      blob would have to be torn apart anyway.
//
// Correctness/performance contract (AGENTS.md §6): HnswCore should match usearch
// within a factor of 2x QPS at identical parameters; #20c (microbench) gates
// that. Any regression beyond that is a user-visible problem for HNSW scans and
// must be attributed to a DuckDB-imposed cost (e.g. block pinning, metric
// indirection) before this task can close.
//
// Algorithmic reference: Malkov & Yashunin, "Efficient and Robust Approximate
// Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs",
// IEEE TPAMI 2018. SelectNeighborsHeuristic = Algorithm 4 of that paper.
// ---------------------------------------------------------------------------

struct HnswCoreParams {
	idx_t dim;
	// Graph degree. Layer >0 uses `m`, layer 0 uses `m0` (classically m0 = 2m).
	uint16_t m = 16;
	uint16_t m0 = 32;
	// Candidate-list width at construction and search time.
	uint16_t ef_construction = 100;
	uint16_t ef_search = 64;
	// Hard cap on level. We grow a node up to this level; any heavier tail
	// would need allocator churn without measurable recall improvement.
	uint8_t max_level = 8;
	uint64_t seed = 0xD00DB1;

	// Size of a node payload at `level` (top layer this node participates
	// in) given a quantizer code size. Exposed so callers can reason about
	// per-node memory before constructing.
	idx_t NodeSize(idx_t code_size, uint8_t level) const;
};

class HnswCore {
public:
	HnswCore(HnswCoreParams params, Quantizer &quantizer, IndexBlockStore &store);

	// Insert a full-precision vector under the given row_id. Returns the node's
	// BlockId, which is how the graph references it internally. Caller is
	// responsible for mapping row_id → BlockId if they need external lookup.
	BlockId Insert(int64_t row_id, const float *vec);

	// Approximate kNN search. Returns up to `k` row_ids sorted by ascending
	// estimator distance. `query_preproc` must be the quantizer's preprocessed
	// query (QueryWorkspaceSize() floats).
	struct Candidate {
		int64_t row_id;
		float distance;
	};
	vector<Candidate> Search(const float *query_preproc, idx_t k, idx_t ef) const;

	// Introspection for #20c perf breakdown and for tests.
	idx_t Size() const {
		return size_;
	}
	idx_t MaxLevel() const {
		return max_level_seen_;
	}
	const HnswCoreParams &Params() const {
		return params_;
	}

	// --- serialization -----------------------------------------------------
	// HnswCore does not own the block store; it serializes only its own state
	// (entry point, max-level, size, node-size-id). The graph edges live in the
	// block store and are persisted as part of IndexBlockStore::GetInfo().
	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

private:
	// ---- node layout helpers ---------------------------------------------
	// A node is a single fixed-size segment in IndexBlockStore, laid out as:
	//   int64_t row_id
	//   uint8_t level                 (this node's top layer)
	//   uint8_t _pad[7]
	//   uint16_t neighbor_count[max_level + 1]
	//   data_t  code[CodeSize()]
	//   BlockId neighbors_l0[m0]
	//   BlockId neighbors_l1[m]
	//   ...
	//   BlockId neighbors_lL[m]       (L = max_level)
	int64_t &NodeRowId(data_ptr_t node) const;
	uint32_t &NodeInternalId(data_ptr_t node) const;
	uint8_t &NodeLevel(data_ptr_t node) const;
	uint16_t *NeighborCounts(data_ptr_t node) const;
	data_ptr_t NodeCode(data_ptr_t node) const;
	BlockId *NeighborArray(data_ptr_t node, uint8_t layer) const;

	idx_t NeighborCapacity(uint8_t layer) const {
		return layer == 0 ? params_.m0 : params_.m;
	}

	uint8_t RandomLevel();

	// ---- core algorithms --------------------------------------------------
	// SearchLayer returns the top-ef candidates to `query_preproc` at `layer`,
	// starting from `entry`. Output is sorted ascending by distance.
	vector<Candidate> SearchLayer(const float *query_preproc, BlockId entry, float entry_dist, idx_t ef,
	                              uint8_t layer) const;

	// Greedy descent: at upper layers we only keep the single nearest node.
	// Returns (best node, best distance).
	std::pair<BlockId, float> GreedySearch(const float *query_preproc, BlockId entry, float entry_dist,
	                                       uint8_t from_layer, uint8_t to_layer) const;

	// Heuristic neighbor selection (Algorithm 4): pick up to M of the given
	// candidates such that no chosen neighbor is closer to another chosen one
	// than to the query. Candidates are consumed in ascending-distance order.
	vector<Candidate> SelectNeighborsHeuristic(const vector<Candidate> &candidates, idx_t M, uint8_t layer) const;

	// Symmetric edge update: connect `new_node` to every member of
	// `selected`. If any of those nodes is over capacity on `layer`, evict
	// that node's farthest existing neighbor (distance measured against
	// `query_preproc`, which is the preprocessed new-node vector — a proxy
	// for d(s, nb) that avoids needing a code-to-code metric).
	void ConnectAndPrune(BlockId new_node, const float *query_preproc, const vector<Candidate> &selected,
	                     uint8_t layer);

	// Cached distance between the query and the code stored in `node`.
	float NodeDistance(const float *query_preproc, BlockId node) const;

private:
	HnswCoreParams params_;
	Quantizer &quantizer_;
	IndexBlockStore &store_;
	// One allocator per level (0..max_level). A node at level L is sized to
	// hold exactly L+1 neighbor arrays (m0 on l0, m on l1..lL) — no wasted
	// slots for levels the node doesn't reach. This is the packing that
	// brings our per-node footprint in line with usearch's.
	vector<NodeSizeId> node_tags_;
	vector<idx_t> node_sizes_;
	idx_t code_size_;

	BlockId entry_;   // null on empty index
	uint8_t max_level_seen_ = 0;
	idx_t size_ = 0;

	// Dense id → visit-epoch table. Epoch-counter visitor: each Search bumps
	// visit_counter_; a node is "visited" iff visit_marks_[internal_id] ==
	// visit_counter_. This replaces std::unordered_set<BlockId.Get()>, which
	// our bench showed spending ~29 ns / distance call on hash + bucket logic.
	// Overflow (counter wraps at 2^32) is handled by resetting the table —
	// that only happens after billions of searches.
	mutable vector<uint32_t> visit_marks_;
	mutable uint32_t visit_counter_ = 0;

	mutable std::mt19937_64 rng_;
};

} // namespace vindex
} // namespace duckdb
