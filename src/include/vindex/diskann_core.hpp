#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/index_block_store.hpp"
#include "vindex/quantizer.hpp"

#include <cstdint>
#include <random>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// DiskAnnCore — single-layer Vamana graph, code stored out-of-band.
//
// Why a separate core (vs. HnswCore + max_level=0)? DiskANN's defining
// feature is that the graph blocks are **not** pinned to RAM: the code lives
// in a contiguous RAM-resident `codes_` array (N × code_size), while the
// graph nodes live in IndexBlockStore segments that the DuckDB
// BufferManager is free to evict. That is what lets the index scale beyond
// RAM. HNSW's inline-code layout can't do that without rewriting the node
// layout — and the algorithmic differences (RobustPrune with alpha,
// unidirectional beam search on a single layer) make a shared core awkward.
//
// Algorithmic reference: Jayaram Subramanya et al., "DiskANN: Fast Accurate
// Billion-point Nearest Neighbor Search on a Single Node", NeurIPS 2019.
//
// RobustPrune: after collecting L candidates via greedy search, pick up to
// R neighbors such that for every chosen neighbor p, no already-chosen
// neighbor p' satisfies alpha · d(p', p) < d(q, p). Alpha > 1 admits more
// long-range edges; the paper recommends alpha ≈ 1.2.
// ---------------------------------------------------------------------------

struct DiskAnnCoreParams {
	idx_t dim = 0;
	uint16_t R = 64;      // graph out-degree
	uint16_t L = 100;     // beam width (used at build and as default at search)
	float alpha = 1.2f;   // RobustPrune relaxation
	uint64_t seed = 0xD15CAFULL;
};

class DiskAnnCore {
public:
	DiskAnnCore(DiskAnnCoreParams params, Quantizer &quantizer, IndexBlockStore &store);

	struct Candidate {
		int64_t row_id;
		float distance;
	};

	// Vamana online insert. Returns the allocated BlockId — caller maps
	// row_id → BlockId for delete bookkeeping.
	BlockId Insert(int64_t row_id, const float *vec);

	// Approximate kNN search. `query_preproc` must come from
	// Quantizer::PreprocessQuery. L_search overrides params_.L when > 0.
	vector<Candidate> Search(const float *query_preproc, idx_t k, idx_t L_search) const;

	idx_t Size() const {
		return size_;
	}
	const DiskAnnCoreParams &Params() const {
		return params_;
	}

	// --- serialization -----------------------------------------------------
	// Layout: {magic:u64, dim:u64, R:u16, L:u16, alpha:f32, seed:u64,
	//          size:u64, entry:u64, code_size:u64, codes:u8[size*code_size]}
	// Graph edges live in IndexBlockStore and persist via its allocator info.
	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

private:
	// Node layout (fixed per-node size = kHeaderBytes + R * sizeof(BlockId)):
	//   int64_t  row_id
	//   uint32_t internal_id   (indexes codes_ and visit_marks_)
	//   uint16_t neighbor_count
	//   uint16_t _pad
	//   BlockId  neighbors[R]
	static constexpr idx_t kHeaderBytes = 16;

	int64_t &NodeRowId(data_ptr_t node) const {
		return *reinterpret_cast<int64_t *>(node);
	}
	uint32_t &NodeInternalId(data_ptr_t node) const {
		return *reinterpret_cast<uint32_t *>(node + 8);
	}
	uint16_t &NeighborCount(data_ptr_t node) const {
		return *reinterpret_cast<uint16_t *>(node + 12);
	}
	BlockId *NeighborArray(data_ptr_t node) const {
		return reinterpret_cast<BlockId *>(node + kHeaderBytes);
	}

	const_data_ptr_t CodeOf(uint32_t internal_id) const {
		return codes_.data() + idx_t(internal_id) * code_size_;
	}
	data_ptr_t MutableCodeOf(uint32_t internal_id) {
		return codes_.data() + idx_t(internal_id) * code_size_;
	}

	float DistanceToCode(const float *query_preproc, uint32_t internal_id) const {
		return quantizer_.EstimateDistance(CodeOf(internal_id), query_preproc);
	}

	// Beam search at build / query time. Returns up to `L` nearest candidates
	// (sorted by ascending distance) to `query_preproc`, starting from the
	// current entry_. Visits edges via the graph nodes in IndexBlockStore;
	// distances are always computed against the codes_ array (not the graph
	// node payload), so cold graph pages can be evicted without losing data.
	vector<Candidate> BeamSearch(const float *query_preproc, idx_t L) const;

	// RobustPrune: select up to R neighbors for `query_preproc` out of
	// `candidates`, using alpha-relaxed diversity. `candidates` is consumed
	// and modified (sorted) but the return is a fresh vector.
	vector<Candidate> RobustPrune(const float *query_preproc, vector<Candidate> candidates, idx_t R,
	                              float alpha) const;

	// Reciprocal edge update: push `new_node` onto every member of
	// `selected`; if the recipient is over R, re-prune its neighbor set.
	// `new_preproc` is the inserted vector's preprocessed query (used to
	// compute d(s, new_node) via the quantizer's asymmetric table).
	void ConnectAndPrune(BlockId new_id, uint32_t new_internal_id, const float *new_preproc,
	                     const vector<Candidate> &selected);

	// Decoded via codebook lookups, not by re-reading the table. Used inside
	// RobustPrune to test d(p', p) without running PreprocessQuery per node.
	float CodeDistance(uint32_t a, uint32_t b) const {
		return quantizer_.CodeDistance(CodeOf(a), CodeOf(b));
	}

	DiskAnnCoreParams params_;
	Quantizer &quantizer_;
	IndexBlockStore &store_;
	NodeSizeId node_tag_ = 0;
	idx_t code_size_ = 0;

	// codes_: contiguous N × code_size_ byte array, authoritative source of
	// codes for distance computations. Graph nodes only hold neighbor ids.
	vector<data_t> codes_;
	// internal_id → BlockId so BeamSearch can pin a node given its index.
	vector<BlockId> node_blocks_;

	BlockId entry_;
	idx_t size_ = 0;

	mutable vector<uint32_t> visit_marks_;
	mutable uint32_t visit_counter_ = 0;
	mutable std::mt19937_64 rng_;
};

} // namespace vindex
} // namespace duckdb
