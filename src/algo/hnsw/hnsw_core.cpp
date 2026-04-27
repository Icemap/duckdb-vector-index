#include "vindex/hnsw_core.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// Node layout:
//   [0]      int64_t  row_id
//   [8]      uint32_t internal_id          (dense 0..N-1; indexes visit_marks_)
//   [12]     uint8_t  level
//   [13..16] padding  (align neighbor_count to 8)
//   [16]     uint16_t neighbor_count[max_level + 1]
//   [..]     data_t   code[code_size]
//   [..]     BlockId  neighbors_l0[m0]
//   [..]     BlockId  neighbors_l1[m]
//   [..]     ...
//   [..]     BlockId  neighbors_lL[m]
// ---------------------------------------------------------------------------

namespace {

constexpr idx_t kHeaderPrefix = 16; // row_id + level + pad

idx_t NeighborBlockSizeForLevel(const HnswCoreParams &p, uint8_t level) {
	// A node at level L has L+1 neighbor arrays: m0 on l0 plus m on each of
	// l1..lL. Contrast with the earlier fixed layout which reserved
	// `m0 + max_level × m` slots on every node regardless of level.
	const idx_t total_slots = p.m0 + idx_t(level) * p.m;
	return total_slots * sizeof(BlockId);
}

idx_t CountsSizeForLevel(uint8_t level) {
	// Only the levels this node participates in (0..level) need a count.
	const idx_t raw = (idx_t(level) + 1) * sizeof(uint16_t);
	return (raw + 7) & ~idx_t(7);
}

} // namespace

idx_t HnswCoreParams::NodeSize(idx_t code_size, uint8_t level) const {
	return kHeaderPrefix + CountsSizeForLevel(level) + code_size + NeighborBlockSizeForLevel(*this, level);
}

int64_t &HnswCore::NodeRowId(data_ptr_t node) const {
	return *reinterpret_cast<int64_t *>(node);
}

uint32_t &HnswCore::NodeInternalId(data_ptr_t node) const {
	return *reinterpret_cast<uint32_t *>(node + 8);
}

uint8_t &HnswCore::NodeLevel(data_ptr_t node) const {
	return *reinterpret_cast<uint8_t *>(node + 12);
}

uint16_t *HnswCore::NeighborCounts(data_ptr_t node) const {
	return reinterpret_cast<uint16_t *>(node + kHeaderPrefix);
}

data_ptr_t HnswCore::NodeCode(data_ptr_t node) const {
	const uint8_t level = *reinterpret_cast<const uint8_t *>(node + 12);
	return node + kHeaderPrefix + CountsSizeForLevel(level);
}

BlockId *HnswCore::NeighborArray(data_ptr_t node, uint8_t layer) const {
	data_ptr_t neigh_base = NodeCode(node) + code_size_;
	idx_t offset = 0;
	if (layer == 0) {
		offset = 0;
	} else {
		offset = params_.m0 + idx_t(layer - 1) * params_.m;
	}
	return reinterpret_cast<BlockId *>(neigh_base) + offset;
}

HnswCore::HnswCore(HnswCoreParams params, Quantizer &quantizer, IndexBlockStore &store)
    : params_(params), quantizer_(quantizer), store_(store), rng_(params.seed) {
	if (params_.max_level == 0) {
		throw InternalException("HnswCore: max_level must be >= 1");
	}
	if (params_.m == 0 || params_.m0 == 0) {
		throw InternalException("HnswCore: m and m0 must be positive");
	}
	if (params_.ef_construction < params_.m) {
		// Classical HNSW requires ef_construction >= M; otherwise we cannot
		// seed the heuristic with enough candidates on each insert.
		params_.ef_construction = params_.m;
	}
	code_size_ = quantizer_.CodeSize();
	// Register one allocator per possible level (0..max_level). Nodes with
	// small levels pay only for the slots they use.
	node_tags_.resize(idx_t(params_.max_level) + 1);
	node_sizes_.resize(idx_t(params_.max_level) + 1);
	for (idx_t L = 0; L <= params_.max_level; L++) {
		node_sizes_[L] = params_.NodeSize(code_size_, uint8_t(L));
		node_tags_[L] = store_.RegisterNodeSize(node_sizes_[L]);
	}
}

uint8_t HnswCore::RandomLevel() {
	// Classical: L = floor(-ln(U) * 1/ln(M)). Capped at max_level.
	std::uniform_real_distribution<double> unif(0.0, 1.0);
	const double u = std::max(unif(rng_), 1e-12);
	const double ml = 1.0 / std::log(double(params_.m));
	const double r = -std::log(u) * ml;
	int level = int(r);
	if (level < 0) {
		level = 0;
	}
	if (level > int(params_.max_level)) {
		level = params_.max_level;
	}
	return uint8_t(level);
}

float HnswCore::NodeDistance(const float *query_preproc, BlockId node_id) const {
	data_ptr_t node = store_.PinFast(node_id);
	return quantizer_.EstimateDistance(NodeCode(node), query_preproc);
}

// ---------------------------------------------------------------------------
// Search primitives
// ---------------------------------------------------------------------------

namespace {

// Min-heap element: nearest first.
struct MinItem {
	float dist;
	BlockId id;
};
struct MinCmp {
	bool operator()(const MinItem &a, const MinItem &b) const {
		return a.dist > b.dist;
	}
};

// Max-heap element (the W candidate set): farthest first, for easy pop of
// the current worst.
struct MaxItem {
	float dist;
	BlockId id;
};
struct MaxCmp {
	bool operator()(const MaxItem &a, const MaxItem &b) const {
		return a.dist < b.dist;
	}
};

} // namespace

vector<HnswCore::Candidate> HnswCore::SearchLayer(const float *query_preproc, BlockId entry, float entry_dist,
                                                  idx_t ef, uint8_t layer) const {
	// Reserved to avoid rehashes. Budget: ef branches × avg-m neighbors, with
	// slack for graph density. Empirically ef*8 keeps load_factor below the
	// rehash threshold through the whole search.
	std::vector<MinItem> cand_buf;
	std::vector<MaxItem> W_buf;
	cand_buf.reserve(ef * 4);
	W_buf.reserve(ef + 1);
	std::priority_queue<MinItem, std::vector<MinItem>, MinCmp> cand(MinCmp(), std::move(cand_buf));
	std::priority_queue<MaxItem, std::vector<MaxItem>, MaxCmp> W(MaxCmp(), std::move(W_buf));

	// Mark the entry as visited. Bump visit_counter_ here so repeated calls
	// within the same Insert() (once per level) each get a fresh epoch.
	visit_counter_++;
	if (visit_counter_ == 0) {
		// Wrapped — zero the table and start at 1.
		std::fill(visit_marks_.begin(), visit_marks_.end(), 0);
		visit_counter_ = 1;
	}
	{
		data_ptr_t entry_ptr = store_.PinFast(entry);
		visit_marks_[NodeInternalId(entry_ptr)] = visit_counter_;
	}
	cand.push({entry_dist, entry});
	W.push({entry_dist, entry});

	const uint32_t vc = visit_counter_;
	uint32_t *marks = visit_marks_.data();
	while (!cand.empty()) {
		const auto c = cand.top();
		cand.pop();
		if (!W.empty() && c.dist > W.top().dist && W.size() >= ef) {
			break;
		}
		data_ptr_t node = store_.PinFast(c.id);
		const uint16_t *counts = NeighborCounts(node);
		const uint16_t n = counts[layer];
		const BlockId *neigh = NeighborArray(node, layer);
		for (uint16_t i = 0; i < n; i++) {
			const BlockId nb = neigh[i];
			data_ptr_t nb_ptr = store_.PinFast(nb);
			const uint32_t iid = NodeInternalId(nb_ptr);
			if (marks[iid] == vc) {
				continue;
			}
			marks[iid] = vc;
			const float d = quantizer_.EstimateDistance(NodeCode(nb_ptr), query_preproc);
			if (W.size() < ef || d < W.top().dist) {
				cand.push({d, nb});
				W.push({d, nb});
				if (W.size() > ef) {
					W.pop();
				}
			}
		}
	}

	vector<Candidate> out;
	out.reserve(W.size());
	while (!W.empty()) {
		auto &t = W.top();
		// row_id is resolved later at Search(); SearchLayer deals in BlockIds.
		// Store it in `row_id` field as a placeholder encoded as the raw
		// BlockId bits — the caller interprets based on context.
		Candidate c;
		c.row_id = int64_t(t.id.Get());
		c.distance = t.dist;
		out.push_back(c);
		W.pop();
	}
	std::reverse(out.begin(), out.end()); // ascending dist
	return out;
}

std::pair<BlockId, float> HnswCore::GreedySearch(const float *query_preproc, BlockId entry, float entry_dist,
                                                 uint8_t from_layer, uint8_t to_layer) const {
	BlockId best = entry;
	float best_d = entry_dist;
	for (int layer = int(from_layer); layer > int(to_layer); layer--) {
		bool changed = true;
		while (changed) {
			changed = false;
			data_ptr_t node = store_.PinFast(best);
			const uint16_t *counts = NeighborCounts(node);
			const uint16_t n = counts[layer];
			const BlockId *neigh = NeighborArray(node, uint8_t(layer));
			for (uint16_t i = 0; i < n; i++) {
				const BlockId nb = neigh[i];
				const float d = NodeDistance(query_preproc, nb);
				if (d < best_d) {
					best_d = d;
					best = nb;
					changed = true;
				}
			}
		}
	}
	return {best, best_d};
}

vector<HnswCore::Candidate> HnswCore::SelectNeighborsHeuristic(const vector<Candidate> &candidates, idx_t M,
                                                                uint8_t /*layer*/) const {
	// Malkov Algorithm 4: iterate candidates in ascending d(q,·) order. Accept
	// candidate c iff every already-accepted neighbor w has d(c,w) >= d(q,c).
	// This spreads chosen neighbors across the space instead of clustering
	// them all near the closest one (which would collapse graph connectivity
	// in a dense region). Candidates arrive sorted by d(q,·) from SearchLayer.
	//
	// Pairwise distances use Quantizer::CodeDistance — O(dim) per pair, which
	// dominates the insert cost. We cache each candidate's code in a local
	// buffer to avoid re-pinning the node store for every comparison.
	vector<Candidate> chosen;
	chosen.reserve(std::min<idx_t>(M, candidates.size()));
	if (candidates.empty() || M == 0) {
		return chosen;
	}

	// Snapshot each candidate's code. We can't hold Pin() across the loop
	// because store_ may relocate buffers on growth; a memcpy snapshot is
	// stable for the lifetime of this call.
	const idx_t cs = code_size_;
	vector<data_t> code_buf(candidates.size() * cs);
	for (idx_t i = 0; i < candidates.size(); i++) {
		BlockId id;
		id.Set(idx_t(candidates[i].row_id));
		data_ptr_t n = store_.PinFast(id);
		std::memcpy(code_buf.data() + i * cs, NodeCode(n), cs);
	}

	// Track which source index each chosen neighbor came from so we can
	// index back into code_buf for the pairwise check.
	vector<idx_t> chosen_src;
	chosen_src.reserve(M);
	for (idx_t i = 0; i < candidates.size(); i++) {
		if (chosen.size() >= M) {
			break;
		}
		const_data_ptr_t ci = code_buf.data() + i * cs;
		bool keep = true;
		for (idx_t j : chosen_src) {
			const_data_ptr_t cj = code_buf.data() + j * cs;
			const float d_ij = quantizer_.CodeDistance(ci, cj);
			// d(q,c) is candidates[i].distance. Reject if any accepted
			// neighbor is strictly closer to c than the query is.
			if (d_ij < candidates[i].distance) {
				keep = false;
				break;
			}
		}
		if (keep) {
			chosen.push_back(candidates[i]);
			chosen_src.push_back(i);
		}
	}
	return chosen;
}

void HnswCore::ConnectAndPrune(BlockId new_node, const float * /*query_preproc*/, const vector<Candidate> &selected,
                               uint8_t layer) {
	{
		data_ptr_t n = store_.PinFast(new_node);
		uint16_t *counts = NeighborCounts(n);
		BlockId *arr = NeighborArray(n, layer);
		const idx_t cap = NeighborCapacity(layer);
		counts[layer] = uint16_t(std::min<idx_t>(selected.size(), cap));
		for (idx_t i = 0; i < counts[layer]; i++) {
			BlockId nb;
			nb.Set(idx_t(selected[i].row_id));
			arr[i] = nb;
		}
	}

	// Snapshot new_node's code once; we'll use it repeatedly to rank
	// (existing_neighbor + new_node) candidates from each s's perspective.
	const idx_t cs = code_size_;
	vector<data_t> new_code(cs);
	{
		data_ptr_t n = store_.PinFast(new_node);
		std::memcpy(new_code.data(), NodeCode(n), cs);
	}

	// Reciprocal edge. When s is at capacity, rerun Algorithm 4 on
	// (existing s-neighbors ∪ {new_node}) ranked by distance-to-s and take
	// the top M. This is the classical Malkov update — it preserves the
	// diversity property that the simple-prune eviction didn't.
	for (const auto &s : selected) {
		BlockId sid;
		sid.Set(idx_t(s.row_id));
		data_ptr_t sn = store_.PinFast(sid);
		uint16_t *counts = NeighborCounts(sn);
		BlockId *arr = NeighborArray(sn, layer);
		const idx_t cap = NeighborCapacity(layer);
		if (counts[layer] < cap) {
			arr[counts[layer]++] = new_node;
			continue;
		}
		// Snapshot s's code, its existing neighbors, and their codes.
		vector<data_t> s_code(cs);
		std::memcpy(s_code.data(), NodeCode(sn), cs);
		vector<BlockId> existing(arr, arr + cap);

		// Build candidate list: each (d(s,w), BlockId, code). d(new,s) = s.distance.
		struct Cand {
			float d;
			BlockId id;
			vector<data_t> code;
		};
		vector<Cand> cands;
		cands.reserve(cap + 1);
		cands.push_back({s.distance, new_node, new_code});
		for (BlockId nb : existing) {
			vector<data_t> nb_code(cs);
			data_ptr_t nb_n = store_.PinFast(nb);
			std::memcpy(nb_code.data(), NodeCode(nb_n), cs);
			const float d = quantizer_.CodeDistance(s_code.data(), nb_code.data());
			cands.push_back({d, nb, std::move(nb_code)});
		}
		std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.d < b.d; });

		// Algorithm 4: accept if no already-accepted w is closer to c than s is.
		vector<BlockId> kept;
		vector<const data_t *> kept_code;
		kept.reserve(cap);
		kept_code.reserve(cap);
		for (const auto &c : cands) {
			if (kept.size() >= cap) {
				break;
			}
			bool ok = true;
			for (const data_t *wc : kept_code) {
				if (quantizer_.CodeDistance(c.code.data(), wc) < c.d) {
					ok = false;
					break;
				}
			}
			if (ok) {
				kept.push_back(c.id);
				kept_code.push_back(c.code.data());
			}
		}

		data_ptr_t sn2 = store_.PinFast(sid);
		uint16_t *counts2 = NeighborCounts(sn2);
		BlockId *arr2 = NeighborArray(sn2, layer);
		counts2[layer] = uint16_t(kept.size());
		for (idx_t i = 0; i < kept.size(); i++) {
			arr2[i] = kept[i];
		}
	}
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

BlockId HnswCore::Insert(int64_t row_id, const float *vec) {
	// Allocate and initialize the new node. We need the level before we
	// know the node size; once chosen, pick the matching per-level allocator.
	const uint8_t level = RandomLevel();
	BlockId new_id = store_.AllocNode(node_tags_[level]);
	const uint32_t internal_id = uint32_t(size_);
	// visit_marks_ is indexed by internal_id; grow it in powers of two to
	// amortize the cost. Fresh slots are left as 0; visit_counter_ starts at
	// 1 so any stale 0 from a freshly-grown slot compares unequal.
	if (internal_id >= visit_marks_.size()) {
		visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, internal_id + 1), 0);
	}
	{
		data_ptr_t n = store_.PinFast(new_id);
		std::memset(n, 0, node_sizes_[level]);
		NodeRowId(n) = row_id;
		NodeInternalId(n) = internal_id;
		NodeLevel(n) = level;
		quantizer_.Encode(vec, NodeCode(n));
	}

	if (size_ == 0) {
		entry_ = new_id;
		max_level_seen_ = level;
		size_ = 1;
		return new_id;
	}

	// Preprocess the query for distance evaluations during the insert.
	vector<float> q(quantizer_.QueryWorkspaceSize());
	quantizer_.PreprocessQuery(vec, q.data());

	// Greedy descent from max_level_seen_ down to level+1.
	BlockId cur = entry_;
	float cur_d = NodeDistance(q.data(), cur);
	if (max_level_seen_ > level) {
		const auto best = GreedySearch(q.data(), cur, cur_d, max_level_seen_, level);
		cur = best.first;
		cur_d = best.second;
	}

	// From level down to 0: SearchLayer, SelectNeighbors, Connect.
	for (int L = int(std::min<uint8_t>(level, max_level_seen_)); L >= 0; L--) {
		auto neighbors =
		    SearchLayer(q.data(), cur, cur_d, params_.ef_construction, uint8_t(L));
		const idx_t M = (L == 0) ? params_.m0 : params_.m;
		auto selected = SelectNeighborsHeuristic(neighbors, M, uint8_t(L));
		ConnectAndPrune(new_id, q.data(), selected, uint8_t(L));
		if (!neighbors.empty()) {
			BlockId best;
			best.Set(idx_t(neighbors.front().row_id));
			cur = best;
			cur_d = neighbors.front().distance;
		}
	}

	if (level > max_level_seen_) {
		max_level_seen_ = level;
		entry_ = new_id;
	}
	size_++;
	return new_id;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

vector<HnswCore::Candidate> HnswCore::Search(const float *query_preproc, idx_t k, idx_t ef) const {
	if (size_ == 0) {
		return {};
	}
	if (ef < k) {
		ef = k;
	}

	float entry_d = NodeDistance(query_preproc, entry_);
	BlockId cur = entry_;
	float cur_d = entry_d;
	if (max_level_seen_ > 0) {
		const auto best = GreedySearch(query_preproc, cur, cur_d, max_level_seen_, 0);
		cur = best.first;
		cur_d = best.second;
	}
	auto cands = SearchLayer(query_preproc, cur, cur_d, ef, 0);
	if (cands.size() > k) {
		cands.resize(k);
	}
	for (auto &c : cands) {
		BlockId id;
		id.Set(idx_t(c.row_id));
		data_ptr_t n = store_.PinFast(id);
		c.row_id = NodeRowId(n);
	}
	return cands;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void HnswCore::SerializeState(vector<data_t> &out) const {
	// Layout: {entry:u64, max_level_seen:u8, size:u64, m:u16, m0:u16,
	//          ef_c:u16, ef_s:u16, max_level:u8, dim:u64, seed:u64}
	out.resize(sizeof(uint64_t) * 4 + sizeof(uint16_t) * 4 + sizeof(uint8_t) * 2);
	idx_t off = 0;
	const uint64_t entry = entry_.Get();
	std::memcpy(out.data() + off, &entry, sizeof(entry));
	off += sizeof(entry);
	out[off++] = max_level_seen_;
	const uint64_t sz = size_;
	std::memcpy(out.data() + off, &sz, sizeof(sz));
	off += sizeof(sz);
	std::memcpy(out.data() + off, &params_.m, sizeof(params_.m));
	off += sizeof(params_.m);
	std::memcpy(out.data() + off, &params_.m0, sizeof(params_.m0));
	off += sizeof(params_.m0);
	std::memcpy(out.data() + off, &params_.ef_construction, sizeof(params_.ef_construction));
	off += sizeof(params_.ef_construction);
	std::memcpy(out.data() + off, &params_.ef_search, sizeof(params_.ef_search));
	off += sizeof(params_.ef_search);
	out[off++] = params_.max_level;
	const uint64_t dim = params_.dim;
	std::memcpy(out.data() + off, &dim, sizeof(dim));
	off += sizeof(dim);
	std::memcpy(out.data() + off, &params_.seed, sizeof(params_.seed));
	off += sizeof(params_.seed);
	out.resize(off);
}

void HnswCore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const idx_t expect = sizeof(uint64_t) * 4 + sizeof(uint16_t) * 4 + sizeof(uint8_t) * 2;
	if (size < expect) {
		throw InternalException("HnswCore::DeserializeState: blob too small");
	}
	idx_t off = 0;
	uint64_t entry;
	std::memcpy(&entry, in + off, sizeof(entry));
	off += sizeof(entry);
	entry_.Set(entry);
	max_level_seen_ = in[off++];
	uint64_t sz;
	std::memcpy(&sz, in + off, sizeof(sz));
	off += sizeof(sz);
	size_ = sz;
	// Resize the visit table to match. The stored internal_ids are in
	// [0, size_); we leave visit_counter_ at 0 so the first post-deserialize
	// Search bumps it to 1 and every slot compares unequal.
	visit_marks_.assign(size_, 0);
	visit_counter_ = 0;
	uint16_t m, m0, efc, efs;
	std::memcpy(&m, in + off, sizeof(m));
	off += sizeof(m);
	std::memcpy(&m0, in + off, sizeof(m0));
	off += sizeof(m0);
	std::memcpy(&efc, in + off, sizeof(efc));
	off += sizeof(efc);
	std::memcpy(&efs, in + off, sizeof(efs));
	off += sizeof(efs);
	const uint8_t max_lvl = in[off++];
	uint64_t dim;
	std::memcpy(&dim, in + off, sizeof(dim));
	off += sizeof(dim);
	uint64_t seed;
	std::memcpy(&seed, in + off, sizeof(seed));
	off += sizeof(seed);
	// Sanity: hyperparams must match the ones used at construction.
	if (m != params_.m || m0 != params_.m0 || efc != params_.ef_construction || efs != params_.ef_search ||
	    max_lvl != params_.max_level || dim != params_.dim) {
		throw InternalException("HnswCore::DeserializeState: hyperparameter mismatch");
	}
	params_.seed = seed;
}

} // namespace vindex
} // namespace duckdb
