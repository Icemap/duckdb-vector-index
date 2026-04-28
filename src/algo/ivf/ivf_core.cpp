#include "vindex/ivf_core.hpp"

#include "algo/ivf/kmeans.hpp"

#include "duckdb/common/exception.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <algorithm>
#include <cstring>
#include <queue>

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
		throw InternalException("IvfCore: state stream truncated");
	}
	T v;
	std::memcpy(&v, cur, sizeof(T));
	cur += sizeof(T);
	return v;
}

// State stream layout (little-endian):
//   u64 magic            ("VNDXIVF1")
//   u64 nlist
//   u64 dim
//   u8  trained
//   f32 centroids[nlist * dim]
//   for each list: { u64 count, u8[] bytes (count * entry_size) }
//   u64 inserted
constexpr uint64_t kStateMagicV1 = 0x315656494E444E56ULL; // "VNDXIVF1"

} // namespace

IvfCore::IvfCore(IvfCoreParams params, Quantizer &quantizer, IndexBlockStore &store)
    : params_(params), quantizer_(quantizer), store_(store) {
	if (params_.nlist == 0) {
		throw InternalException("IvfCore: nlist must be >= 1");
	}
	if (params_.dim == 0) {
		throw InternalException("IvfCore: dim must be >= 1");
	}
	if (params_.nprobe == 0) {
		params_.nprobe = 1;
	}
	if (params_.nprobe > params_.nlist) {
		params_.nprobe = params_.nlist;
	}
	code_size_ = quantizer_.CodeSize();

	centroids_.assign(params_.nlist * params_.dim, 0.0f);
	postings_.assign(params_.nlist, vector<data_t> {});
	postings_count_.assign(params_.nlist, 0);
	(void)store_; // reserved — streaming persistence currently routes through IvfIndex.
}

IvfCore::~IvfCore() = default;

void IvfCore::Train(const float *samples, idx_t n) {
	if (trained_) {
		throw InternalException("IvfCore::Train called twice");
	}
	ivf::KMeansPlusPlus(samples, n, params_.dim, params_.nlist, params_.seed, params_.kmeans_iters,
	                    centroids_.data());
	trained_ = true;
}

idx_t IvfCore::AssignCentroid(const float *vec) const {
	float best = std::numeric_limits<float>::infinity();
	idx_t best_c = 0;
	for (idx_t c = 0; c < params_.nlist; c++) {
		const float d = L2sq(vec, centroids_.data() + c * params_.dim, params_.dim);
		if (d < best) {
			best = d;
			best_c = c;
		}
	}
	return best_c;
}

idx_t IvfCore::Insert(int64_t row_id, const float *vec) {
	if (!trained_) {
		throw InternalException("IvfCore::Insert called before Train");
	}
	const idx_t c = AssignCentroid(vec);

	const idx_t entry_size = EntrySize();
	auto &bytes = postings_[c];
	const idx_t off = bytes.size();
	bytes.resize(off + entry_size);
	std::memcpy(bytes.data() + off, &row_id, sizeof(int64_t));
	quantizer_.Encode(vec, bytes.data() + off + sizeof(int64_t));
	postings_count_[c]++;
	inserted_++;
	return c;
}

vector<IvfCore::Candidate> IvfCore::Search(const float *query_exact, const float *query_preproc, idx_t limit,
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

void IvfCore::SerializeState(vector<data_t> &out) const {
	Append<uint64_t>(out, kStateMagicV1);
	Append<uint64_t>(out, uint64_t(params_.nlist));
	Append<uint64_t>(out, uint64_t(params_.dim));
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

void IvfCore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const_data_ptr_t cur = in;
	const_data_ptr_t end = in + size;

	const auto magic = Consume<uint64_t>(cur, end);
	if (magic != kStateMagicV1) {
		throw InternalException("IvfCore: unrecognized state stream (magic mismatch)");
	}
	const auto nlist = Consume<uint64_t>(cur, end);
	const auto dim = Consume<uint64_t>(cur, end);
	// `dim` is fixed by the column type and must always match; `nlist` on the
	// other hand may not be reflected in reloaded WITH-options (DuckDB re-
	// instantiates BoundIndex from the catalog with the original options, but
	// we treat the persisted blob as authoritative for the structural params
	// that affect layout).
	if (dim != params_.dim) {
		throw InternalException("IvfCore: state stream dim mismatch (stored=%llu, runtime=%llu)",
		                        (unsigned long long)dim, (unsigned long long)params_.dim);
	}
	if (nlist != params_.nlist) {
		// Re-shape the in-memory structures to match what was persisted.
		params_.nlist = idx_t(nlist);
		centroids_.assign(params_.nlist * params_.dim, 0.0f);
		postings_.assign(params_.nlist, vector<data_t> {});
		postings_count_.assign(params_.nlist, 0);
		if (params_.nprobe > params_.nlist) {
			params_.nprobe = params_.nlist;
		}
	}
	trained_ = Consume<uint8_t>(cur, end) != 0;

	const idx_t cbytes = centroids_.size() * sizeof(float);
	if (cur + cbytes > end) {
		throw InternalException("IvfCore: state stream truncated at centroids");
	}
	std::memcpy(centroids_.data(), cur, cbytes);
	cur += cbytes;

	const idx_t entry_size = EntrySize();
	for (idx_t c = 0; c < params_.nlist; c++) {
		const auto count = Consume<uint64_t>(cur, end);
		const idx_t bytes = count * entry_size;
		if (cur + bytes > end) {
			throw InternalException("IvfCore: state stream truncated in posting list %llu", (unsigned long long)c);
		}
		postings_count_[c] = count;
		postings_[c].assign(cur, cur + bytes);
		cur += bytes;
	}
	inserted_ = idx_t(Consume<uint64_t>(cur, end));
}

} // namespace vindex
} // namespace duckdb
