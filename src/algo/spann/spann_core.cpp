#include "vindex/spann_core.hpp"

#include "algo/ivf/kmeans.hpp"

#include "duckdb/common/exception.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <unordered_set>

namespace duckdb {
namespace vindex {

namespace {

inline float L2sq(const float *a, const float *b, idx_t dim) {
	simsimd_distance_t d = 0;
	simsimd_l2sq_f32(reinterpret_cast<const simsimd_f32_t *>(a), reinterpret_cast<const simsimd_f32_t *>(b), dim, &d);
	return float(d);
}

template <typename T>
void Append(vector<data_t> &out, const T &v) {
	const auto *p = reinterpret_cast<const data_t *>(&v);
	out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T Consume(const_data_ptr_t &cur, const_data_ptr_t end) {
	if (cur + sizeof(T) > end) {
		throw InternalException("SpannCore: state stream truncated");
	}
	T v;
	std::memcpy(&v, cur, sizeof(T));
	cur += sizeof(T);
	return v;
}

// State stream layout (little-endian):
//   u64 magic            ("VNDXSPN1")
//   u64 nlist
//   u64 dim
//   u64 replica_count
//   f32 closure_factor
//   u8  trained
//   f32 centroids[nlist * dim]
//   for each list: { u64 count, u8[] bytes (count * entry_size) }
//   u64 inserted
constexpr uint64_t kStateMagicV1 = 0x314E50534E444E56ULL; // "VNDXSPN1"

} // namespace

SpannCore::SpannCore(SpannCoreParams params, Quantizer &quantizer, IndexBlockStore &store)
    : params_(params), quantizer_(quantizer), store_(store) {
	if (params_.nlist == 0) {
		throw InternalException("SpannCore: nlist must be >= 1");
	}
	if (params_.dim == 0) {
		throw InternalException("SpannCore: dim must be >= 1");
	}
	if (params_.nprobe == 0) {
		params_.nprobe = 1;
	}
	if (params_.nprobe > params_.nlist) {
		params_.nprobe = params_.nlist;
	}
	if (params_.replica_count == 0) {
		params_.replica_count = 1;
	}
	if (params_.replica_count > params_.nlist) {
		params_.replica_count = params_.nlist;
	}
	if (!(params_.closure_factor >= 1.0f)) {
		// closure_factor < 1 would exclude the primary centroid itself; clamp.
		params_.closure_factor = 1.0f;
	}
	code_size_ = quantizer_.CodeSize();

	centroids_.assign(params_.nlist * params_.dim, 0.0f);
	postings_.assign(params_.nlist, vector<data_t> {});
	postings_count_.assign(params_.nlist, 0);
	(void)store_;
}

SpannCore::~SpannCore() = default;

void SpannCore::Train(const float *samples, idx_t n) {
	if (trained_) {
		throw InternalException("SpannCore::Train called twice");
	}
	ivf::KMeansPlusPlus(samples, n, params_.dim, params_.nlist, params_.seed, params_.kmeans_iters,
	                    centroids_.data());
	trained_ = true;
}

vector<idx_t> SpannCore::ChooseCentroids(const float *vec) const {
	// Score every centroid, sort by distance, then keep the closure set.
	vector<std::pair<float, idx_t>> scored;
	scored.reserve(params_.nlist);
	for (idx_t c = 0; c < params_.nlist; c++) {
		scored.emplace_back(L2sq(vec, centroids_.data() + c * params_.dim, params_.dim), c);
	}
	const idx_t cap = std::min<idx_t>(params_.replica_count, scored.size());
	std::partial_sort(scored.begin(), scored.begin() + cap, scored.end(),
	                  [](const std::pair<float, idx_t> &a, const std::pair<float, idx_t> &b) {
		                  return a.first < b.first;
	                  });

	vector<idx_t> out;
	out.reserve(cap);
	if (cap == 0) {
		return out;
	}
	// Always take the primary (closest) centroid — that is the invariant the
	// dedup / Delete bookkeeping relies on.
	out.push_back(scored[0].second);
	const float d_best = scored[0].first;
	const float cutoff = params_.closure_factor * d_best;
	for (idx_t i = 1; i < cap; i++) {
		// Inclusive cutoff so exact ties survive; the rare `d_best == 0`
		// case still picks only the primary (later replicas would all need
		// d ≤ 0 which is impossible for L2sq > 0).
		if (scored[i].first <= cutoff) {
			out.push_back(scored[i].second);
		} else {
			break; // scored is in ascending order — the rest are out of range.
		}
	}
	return out;
}

idx_t SpannCore::Insert(int64_t row_id, const float *vec) {
	if (!trained_) {
		throw InternalException("SpannCore::Insert called before Train");
	}
	const auto cells = ChooseCentroids(vec);
	D_ASSERT(!cells.empty());

	// Encode once — all replicas share the same code. Writing distinct codes
	// per replica (e.g. residual-relative encodings) is possible but not in
	// the SPANN paper, and not worth the quantizer plumbing complexity.
	vector<data_t> code_buf(code_size_);
	quantizer_.Encode(vec, code_buf.data());

	const idx_t entry_size = EntrySize();
	for (idx_t c : cells) {
		auto &bytes = postings_[c];
		const idx_t off = bytes.size();
		bytes.resize(off + entry_size);
		std::memcpy(bytes.data() + off, &row_id, sizeof(int64_t));
		std::memcpy(bytes.data() + off + sizeof(int64_t), code_buf.data(), code_size_);
		postings_count_[c]++;
	}
	inserted_++;
	return cells.front();
}

idx_t SpannCore::Entries() const {
	idx_t total = 0;
	for (auto c : postings_count_) {
		total += idx_t(c);
	}
	return total;
}

vector<SpannCore::Candidate> SpannCore::Search(const float *query_exact, const float *query_preproc, idx_t limit,
                                               idx_t nprobe_override) const {
	vector<Candidate> out;
	if (limit == 0 || inserted_ == 0) {
		return out;
	}
	const idx_t nprobe = nprobe_override > 0 ? std::min<idx_t>(nprobe_override, params_.nlist) : params_.nprobe;

	vector<std::pair<float, idx_t>> centroid_dists;
	centroid_dists.reserve(params_.nlist);
	for (idx_t c = 0; c < params_.nlist; c++) {
		centroid_dists.emplace_back(L2sq(query_exact, centroids_.data() + c * params_.dim, params_.dim), c);
	}
	const idx_t take = std::min<idx_t>(nprobe, centroid_dists.size());
	std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + take, centroid_dists.end(),
	                  [](const std::pair<float, idx_t> &a, const std::pair<float, idx_t> &b) {
		                  return a.first < b.first;
	                  });

	struct HeapEntry {
		float dist;
		int64_t row_id;
		bool operator<(const HeapEntry &o) const {
			return dist < o.dist;
		}
	};
	std::priority_queue<HeapEntry> heap;

	// Dedup by row_id — replica writes mean the same row can surface from
	// multiple probed lists. The paper's §3.2 prescribes the same.
	std::unordered_set<int64_t> seen;
	seen.reserve(limit * 4);

	const idx_t entry_size = EntrySize();
	for (idx_t pi = 0; pi < take; pi++) {
		const idx_t c = centroid_dists[pi].second;
		const auto &bytes = postings_[c];
		const idx_t count = postings_count_[c];
		if (count == 0) {
			continue;
		}
		for (idx_t i = 0; i < count; i++) {
			const_data_ptr_t entry = bytes.data() + i * entry_size;
			int64_t row_id;
			std::memcpy(&row_id, entry, sizeof(int64_t));
			if (!seen.insert(row_id).second) {
				continue;
			}
			const float d = quantizer_.EstimateDistance(entry + sizeof(int64_t), query_preproc);
			if (heap.size() < limit) {
				heap.push({d, row_id});
			} else if (d < heap.top().dist) {
				heap.pop();
				heap.push({d, row_id});
			}
		}
	}

	out.reserve(heap.size());
	while (!heap.empty()) {
		out.push_back({heap.top().dist, heap.top().row_id});
		heap.pop();
	}
	std::reverse(out.begin(), out.end());
	return out;
}

void SpannCore::SerializeState(vector<data_t> &out) const {
	Append<uint64_t>(out, kStateMagicV1);
	Append<uint64_t>(out, uint64_t(params_.nlist));
	Append<uint64_t>(out, uint64_t(params_.dim));
	Append<uint64_t>(out, uint64_t(params_.replica_count));
	Append<float>(out, params_.closure_factor);
	Append<uint8_t>(out, trained_ ? 1 : 0);

	const auto *cbytes = reinterpret_cast<const data_t *>(centroids_.data());
	out.insert(out.end(), cbytes, cbytes + centroids_.size() * sizeof(float));

	for (idx_t c = 0; c < params_.nlist; c++) {
		Append<uint64_t>(out, postings_count_[c]);
		const auto &bytes = postings_[c];
		out.insert(out.end(), bytes.begin(), bytes.end());
	}
	Append<uint64_t>(out, uint64_t(inserted_));
}

void SpannCore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const_data_ptr_t cur = in;
	const_data_ptr_t end = in + size;

	const auto magic = Consume<uint64_t>(cur, end);
	if (magic != kStateMagicV1) {
		throw InternalException("SpannCore: unrecognized state stream (magic mismatch)");
	}
	const auto nlist = Consume<uint64_t>(cur, end);
	const auto dim = Consume<uint64_t>(cur, end);
	const auto replica_count = Consume<uint64_t>(cur, end);
	const auto closure_factor = Consume<float>(cur, end);
	if (dim != params_.dim) {
		throw InternalException("SpannCore: state stream dim mismatch (stored=%llu, runtime=%llu)",
		                        (unsigned long long)dim, (unsigned long long)params_.dim);
	}
	if (nlist != params_.nlist) {
		params_.nlist = idx_t(nlist);
		centroids_.assign(params_.nlist * params_.dim, 0.0f);
		postings_.assign(params_.nlist, vector<data_t> {});
		postings_count_.assign(params_.nlist, 0);
		if (params_.nprobe > params_.nlist) {
			params_.nprobe = params_.nlist;
		}
	}
	params_.replica_count = idx_t(replica_count);
	params_.closure_factor = closure_factor;
	if (params_.replica_count == 0) {
		params_.replica_count = 1;
	}
	if (params_.replica_count > params_.nlist) {
		params_.replica_count = params_.nlist;
	}
	if (!(params_.closure_factor >= 1.0f)) {
		params_.closure_factor = 1.0f;
	}
	trained_ = Consume<uint8_t>(cur, end) != 0;

	const idx_t cbytes = centroids_.size() * sizeof(float);
	if (cur + cbytes > end) {
		throw InternalException("SpannCore: state stream truncated at centroids");
	}
	std::memcpy(centroids_.data(), cur, cbytes);
	cur += cbytes;

	const idx_t entry_size = EntrySize();
	for (idx_t c = 0; c < params_.nlist; c++) {
		const auto count = Consume<uint64_t>(cur, end);
		const idx_t bytes = count * entry_size;
		if (cur + bytes > end) {
			throw InternalException("SpannCore: state stream truncated in posting list %llu", (unsigned long long)c);
		}
		postings_count_[c] = count;
		postings_[c].assign(cur, cur + bytes);
		cur += bytes;
	}
	inserted_ = idx_t(Consume<uint64_t>(cur, end));
}

} // namespace vindex
} // namespace duckdb
