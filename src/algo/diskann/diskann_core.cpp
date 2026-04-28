#include "vindex/diskann_core.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cstring>
#include <queue>

namespace duckdb {
namespace vindex {

namespace {

constexpr uint64_t kStateMagicV1 = 0x3156534B444E4456ULL; // "VDNDKSV1" (magic tag)

// Min-heap item for beam-search frontier (pop closest first).
struct FrontierItem {
	float dist;
	uint32_t internal_id;
};
struct FrontierCmp {
	bool operator()(const FrontierItem &a, const FrontierItem &b) const {
		return a.dist > b.dist;
	}
};

// Max-heap item for the working candidate set W (pop farthest first).
struct WorkingItem {
	float dist;
	uint32_t internal_id;
};
struct WorkingCmp {
	bool operator()(const WorkingItem &a, const WorkingItem &b) const {
		return a.dist < b.dist;
	}
};

} // namespace

DiskAnnCore::DiskAnnCore(DiskAnnCoreParams params, Quantizer &quantizer, IndexBlockStore &store)
    : params_(params), quantizer_(quantizer), store_(store), rng_(params.seed) {
	if (params_.R == 0) {
		throw InternalException("DiskAnnCore: R must be >= 1");
	}
	if (params_.L < params_.R) {
		params_.L = params_.R;
	}
	if (params_.alpha < 1.0f) {
		params_.alpha = 1.0f;
	}
	code_size_ = quantizer_.CodeSize();
	const idx_t node_size = kHeaderBytes + idx_t(params_.R) * sizeof(BlockId);
	node_tag_ = store_.RegisterNodeSize(node_size);
}

// ---------------------------------------------------------------------------
// BeamSearch
// ---------------------------------------------------------------------------

vector<DiskAnnCore::Candidate> DiskAnnCore::BeamSearch(const float *query_preproc, idx_t L) const {
	vector<Candidate> out;
	if (size_ == 0) {
		return out;
	}
	if (L == 0) {
		L = 1;
	}

	// Bump visit epoch. Overflow after 2^32 searches — rezero the table.
	visit_counter_++;
	if (visit_counter_ == 0) {
		std::fill(visit_marks_.begin(), visit_marks_.end(), 0);
		visit_counter_ = 1;
	}
	const uint32_t vc = visit_counter_;

	std::priority_queue<FrontierItem, std::vector<FrontierItem>, FrontierCmp> frontier;
	std::priority_queue<WorkingItem, std::vector<WorkingItem>, WorkingCmp> W;

	const uint32_t entry_internal = ([&]() {
		data_ptr_t e = store_.PinFast(entry_);
		return NodeInternalId(e);
	})();
	const float entry_dist = DistanceToCode(query_preproc, entry_internal);
	if (entry_internal >= visit_marks_.size()) {
		visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, size_t(entry_internal) + 1), 0);
	}
	visit_marks_[entry_internal] = vc;
	frontier.push({entry_dist, entry_internal});
	W.push({entry_dist, entry_internal});

	while (!frontier.empty()) {
		const auto best = frontier.top();
		frontier.pop();
		if (W.size() >= L && best.dist > W.top().dist) {
			break;
		}
		// Visit neighbors of `best`. node_blocks_[best.internal_id] may alias
		// a buffer DuckDB has evicted; PinFast brings it back in.
		BlockId nid = node_blocks_[best.internal_id];
		data_ptr_t node = store_.PinFast(nid);
		const uint16_t n = NeighborCount(node);
		const BlockId *neigh = NeighborArray(node);
		for (uint16_t i = 0; i < n; i++) {
			BlockId nb_block = neigh[i];
			data_ptr_t nb_node = store_.PinFast(nb_block);
			const uint32_t nb_internal = NodeInternalId(nb_node);
			if (nb_internal >= visit_marks_.size()) {
				visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, size_t(nb_internal) + 1), 0);
			}
			if (visit_marks_[nb_internal] == vc) {
				continue;
			}
			visit_marks_[nb_internal] = vc;
			const float d = DistanceToCode(query_preproc, nb_internal);
			if (W.size() < L || d < W.top().dist) {
				frontier.push({d, nb_internal});
				W.push({d, nb_internal});
				if (W.size() > L) {
					W.pop();
				}
			}
		}
	}

	out.reserve(W.size());
	while (!W.empty()) {
		const auto &t = W.top();
		Candidate c;
		c.row_id = int64_t(t.internal_id); // caller resolves internal_id → row_id
		c.distance = t.dist;
		out.push_back(c);
		W.pop();
	}
	std::reverse(out.begin(), out.end()); // ascending
	return out;
}

// ---------------------------------------------------------------------------
// RobustPrune
// ---------------------------------------------------------------------------

vector<DiskAnnCore::Candidate> DiskAnnCore::RobustPrune(const float * /*query_preproc*/,
                                                       vector<Candidate> candidates, idx_t R,
                                                       float alpha) const {
	// Algorithm: sort by d(q, ·) ascending. Repeatedly pop the nearest, add it
	// to the output, and from the remaining candidates drop every p' such that
	// alpha · d(chosen, p') ≤ d(q, p'). This keeps a geometrically spread
	// neighbor set — the diskann paper's main lemma is that this gives
	// bounded-degree navigable graphs on arbitrary datasets.
	std::sort(candidates.begin(), candidates.end(),
	          [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
	vector<Candidate> kept;
	kept.reserve(R);
	while (!candidates.empty() && kept.size() < R) {
		Candidate p = candidates.front();
		kept.push_back(p);
		// Partition remaining candidates: keep those whose distance to p is
		// "too far" relative to their d(q, ·). Runs in O(|candidates|) per
		// chosen — cost is bounded by R × L ≈ 6400 CodeDistance calls.
		vector<Candidate> remaining;
		remaining.reserve(candidates.size());
		for (idx_t i = 1; i < candidates.size(); i++) {
			const auto &pp = candidates[i];
			const float d_pp = CodeDistance(uint32_t(p.row_id), uint32_t(pp.row_id));
			if (alpha * d_pp > pp.distance) {
				remaining.push_back(pp);
			}
		}
		candidates = std::move(remaining);
	}
	return kept;
}

// ---------------------------------------------------------------------------
// ConnectAndPrune
// ---------------------------------------------------------------------------

void DiskAnnCore::ConnectAndPrune(BlockId new_id, uint32_t new_internal_id, const float *new_preproc,
                                  const vector<Candidate> &selected) {
	// Write the forward edges on the new node.
	{
		data_ptr_t n = store_.PinFast(new_id);
		uint16_t &cnt = NeighborCount(n);
		BlockId *arr = NeighborArray(n);
		cnt = uint16_t(std::min<idx_t>(selected.size(), params_.R));
		for (idx_t i = 0; i < cnt; i++) {
			arr[i] = node_blocks_[uint32_t(selected[i].row_id)];
		}
	}

	// Reciprocal edges. For each chosen neighbor s, push new_node onto s's
	// neighbor list. If that overflows R, re-prune s.
	for (const auto &s : selected) {
		const uint32_t s_internal = uint32_t(s.row_id);
		BlockId sid = node_blocks_[s_internal];
		data_ptr_t sn = store_.PinFast(sid);
		uint16_t &cnt = NeighborCount(sn);
		BlockId *arr = NeighborArray(sn);
		if (cnt < params_.R) {
			arr[cnt++] = new_id;
			continue;
		}
		// Collect s's current neighbors + new_node, rank by d(s, ·) via
		// CodeDistance, then RobustPrune back to R.
		vector<Candidate> cand;
		cand.reserve(cnt + 1);
		cand.push_back({int64_t(new_internal_id), CodeDistance(s_internal, new_internal_id)});
		for (uint16_t i = 0; i < cnt; i++) {
			BlockId nb = arr[i];
			data_ptr_t nb_node = store_.PinFast(nb);
			const uint32_t nb_internal = NodeInternalId(nb_node);
			cand.push_back({int64_t(nb_internal), CodeDistance(s_internal, nb_internal)});
		}
		auto kept = RobustPrune(new_preproc, std::move(cand), params_.R, params_.alpha);

		data_ptr_t sn2 = store_.PinFast(sid);
		uint16_t &cnt2 = NeighborCount(sn2);
		BlockId *arr2 = NeighborArray(sn2);
		cnt2 = uint16_t(kept.size());
		for (idx_t i = 0; i < kept.size(); i++) {
			arr2[i] = node_blocks_[uint32_t(kept[i].row_id)];
		}
	}
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

BlockId DiskAnnCore::Insert(int64_t row_id, const float *vec) {
	const uint32_t internal_id = uint32_t(size_);

	// 1) Materialize the code.
	codes_.resize(codes_.size() + code_size_);
	quantizer_.Encode(vec, MutableCodeOf(internal_id));

	// 2) Allocate and zero the graph node.
	BlockId new_id = store_.AllocNode(node_tag_);
	{
		data_ptr_t n = store_.PinFast(new_id);
		std::memset(n, 0, kHeaderBytes + idx_t(params_.R) * sizeof(BlockId));
		NodeRowId(n) = row_id;
		NodeInternalId(n) = internal_id;
		NeighborCount(n) = 0;
	}
	node_blocks_.push_back(new_id);
	if (internal_id >= visit_marks_.size()) {
		visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, size_t(internal_id) + 1), 0);
	}

	if (size_ == 0) {
		entry_ = new_id;
		size_ = 1;
		return new_id;
	}

	// 3) Preprocess the query once, reuse through BeamSearch / ConnectAndPrune.
	vector<float> qpre(quantizer_.QueryWorkspaceSize());
	quantizer_.PreprocessQuery(vec, qpre.data());

	auto cands = BeamSearch(qpre.data(), params_.L);
	// BeamSearch returns items whose row_id field is the internal_id. Filter
	// out self (shouldn't appear: the new node hasn't been connected yet).
	auto selected = RobustPrune(qpre.data(), std::move(cands), params_.R, params_.alpha);
	ConnectAndPrune(new_id, internal_id, qpre.data(), selected);

	size_++;
	return new_id;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

vector<DiskAnnCore::Candidate> DiskAnnCore::Search(const float *query_preproc, idx_t k, idx_t L_search) const {
	if (size_ == 0 || k == 0) {
		return {};
	}
	if (L_search == 0) {
		L_search = params_.L;
	}
	if (L_search < k) {
		L_search = k;
	}
	auto cands = BeamSearch(query_preproc, L_search);
	if (cands.size() > k) {
		cands.resize(k);
	}
	// Resolve internal_id → row_id via the graph nodes.
	for (auto &c : cands) {
		BlockId nid = node_blocks_[uint32_t(c.row_id)];
		data_ptr_t node = store_.PinFast(nid);
		c.row_id = NodeRowId(node);
	}
	return cands;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

namespace {

template <typename T>
void Append(vector<data_t> &out, const T &v) {
	const auto *p = reinterpret_cast<const data_t *>(&v);
	out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T Consume(const_data_ptr_t &cur, const_data_ptr_t end) {
	if (cur + sizeof(T) > end) {
		throw InternalException("DiskAnnCore: state stream truncated");
	}
	T v;
	std::memcpy(&v, cur, sizeof(T));
	cur += sizeof(T);
	return v;
}

} // namespace

void DiskAnnCore::SerializeState(vector<data_t> &out) const {
	Append<uint64_t>(out, kStateMagicV1);
	Append<uint64_t>(out, uint64_t(params_.dim));
	Append<uint16_t>(out, params_.R);
	Append<uint16_t>(out, params_.L);
	Append<float>(out, params_.alpha);
	Append<uint64_t>(out, params_.seed);
	Append<uint64_t>(out, uint64_t(size_));
	Append<uint64_t>(out, uint64_t(entry_.Get()));
	Append<uint64_t>(out, uint64_t(code_size_));
	if (!codes_.empty()) {
		out.insert(out.end(), codes_.begin(), codes_.end());
	}
	Append<uint64_t>(out, uint64_t(node_blocks_.size()));
	for (auto &bid : node_blocks_) {
		Append<uint64_t>(out, uint64_t(bid.Get()));
	}
}

void DiskAnnCore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const_data_ptr_t cur = in;
	const_data_ptr_t end = in + size;

	const auto magic = Consume<uint64_t>(cur, end);
	if (magic != kStateMagicV1) {
		throw InternalException("DiskAnnCore: unrecognized state stream (magic mismatch)");
	}
	const auto dim = Consume<uint64_t>(cur, end);
	const auto R = Consume<uint16_t>(cur, end);
	const auto L = Consume<uint16_t>(cur, end);
	const auto alpha = Consume<float>(cur, end);
	const auto seed = Consume<uint64_t>(cur, end);
	if (dim != params_.dim) {
		throw InternalException("DiskAnnCore: dim mismatch (stored=%llu runtime=%llu)",
		                        (unsigned long long)dim, (unsigned long long)params_.dim);
	}
	params_.R = R;
	params_.L = L;
	params_.alpha = alpha;
	params_.seed = seed;

	size_ = idx_t(Consume<uint64_t>(cur, end));
	uint64_t entry_raw = Consume<uint64_t>(cur, end);
	entry_.Set(entry_raw);
	const auto stored_code_size = Consume<uint64_t>(cur, end);
	if (stored_code_size != code_size_) {
		throw InternalException("DiskAnnCore: code_size mismatch (stored=%llu runtime=%llu)",
		                        (unsigned long long)stored_code_size, (unsigned long long)code_size_);
	}
	const idx_t code_bytes = size_ * code_size_;
	if (cur + code_bytes > end) {
		throw InternalException("DiskAnnCore: state stream truncated at codes");
	}
	codes_.assign(cur, cur + code_bytes);
	cur += code_bytes;

	const auto node_count = Consume<uint64_t>(cur, end);
	if (node_count != size_) {
		throw InternalException("DiskAnnCore: node_blocks size mismatch");
	}
	node_blocks_.clear();
	node_blocks_.reserve(node_count);
	for (uint64_t i = 0; i < node_count; i++) {
		BlockId bid;
		bid.Set(Consume<uint64_t>(cur, end));
		node_blocks_.push_back(bid);
	}
	visit_marks_.assign(size_, 0);
	visit_counter_ = 0;
}

} // namespace vindex
} // namespace duckdb
