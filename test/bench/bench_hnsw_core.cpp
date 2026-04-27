// HnswCore standalone microbench. The usearch side-by-side comparison that
// justified M1.6d (dropping usearch) is captured in the README; the numbers
// below regress against HnswCore itself.
//
// Output columns:
//   build_s   : time to insert all N vectors (single-threaded)
//   qps       : queries-per-second at ef_search
//   recall@k  : fraction of brute-force top-k recovered
//   mem_mb    : RSS delta over the build (best-effort; mach task_info on macOS)

#include "duckdb.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "vindex/hnsw_core.hpp"
#include "vindex/index_block_store.hpp"
#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

using duckdb::BlockManager;
using duckdb::DatabaseManager;
using duckdb::DuckDB;
using duckdb::idx_t;
using duckdb::vindex::HnswCore;
using duckdb::vindex::HnswCoreParams;
using duckdb::vindex::IndexBlockStore;
using duckdb::vindex::MetricKind;
using duckdb::vindex::Quantizer;
using duckdb::vindex::QuantizerKind;

namespace {

// Identity quantizer (same trick used in test_hnsw_core.cpp — FlatQuantizer's
// concrete type is `namespace {}` inside flat_quantizer.cpp).
class FlatExact : public Quantizer {
public:
	explicit FlatExact(idx_t dim) : dim_(dim) {
	}
	void Train(const float *, idx_t, idx_t) override {
	}
	void Encode(const float *vec, duckdb::data_ptr_t out) const override {
		std::memcpy(out, vec, dim_ * sizeof(float));
	}
	float EstimateDistance(duckdb::const_data_ptr_t code, const float *q) const override {
		simsimd_distance_t d = 0;
		simsimd_l2sq_f32(reinterpret_cast<const simsimd_f32_t *>(code), q, dim_, &d);
		return float(d);
	}
	float CodeDistance(duckdb::const_data_ptr_t a, duckdb::const_data_ptr_t b) const override {
		simsimd_distance_t d = 0;
		simsimd_l2sq_f32(reinterpret_cast<const simsimd_f32_t *>(a),
		                 reinterpret_cast<const simsimd_f32_t *>(b), dim_, &d);
		return float(d);
	}
	void PreprocessQuery(const float *q, float *out) const override {
		std::memcpy(out, q, dim_ * sizeof(float));
	}
	idx_t CodeSize() const override {
		return dim_ * sizeof(float);
	}
	idx_t QueryWorkspaceSize() const override {
		return dim_;
	}
	MetricKind Metric() const override {
		return MetricKind::L2SQ;
	}
	QuantizerKind Kind() const override {
		return QuantizerKind::FLAT;
	}
	void Serialize(duckdb::vector<duckdb::data_t> &) const override {
	}
	void Deserialize(duckdb::const_data_ptr_t, idx_t) override {
	}

private:
	idx_t dim_;
};

std::vector<float> MakeGaussian(idx_t n, idx_t dim, uint64_t seed) {
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	std::vector<float> v(n * dim);
	for (auto &x : v) {
		x = nd(rng);
	}
	return v;
}

std::vector<std::vector<int64_t>> BruteForce(const float *data, idx_t n, idx_t dim, const float *queries, idx_t nq,
                                              idx_t k) {
	std::vector<std::vector<int64_t>> out(nq);
	std::vector<std::pair<float, int64_t>> scored(n);
	for (idx_t q = 0; q < nq; q++) {
		for (idx_t i = 0; i < n; i++) {
			float acc = 0.0f;
			for (idx_t j = 0; j < dim; j++) {
				const float d = data[i * dim + j] - queries[q * dim + j];
				acc += d * d;
			}
			scored[i] = {acc, int64_t(i)};
		}
		std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
		out[q].resize(k);
		for (idx_t i = 0; i < k; i++) {
			out[q][i] = scored[i].second;
		}
	}
	return out;
}

double ResidentMB() {
#ifdef __APPLE__
	mach_task_basic_info info;
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
	    KERN_SUCCESS) {
		return double(info.resident_size) / (1024.0 * 1024.0);
	}
	return 0.0;
#elif defined(__linux__)
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	return double(ru.ru_maxrss) / 1024.0;
#else
	return 0.0;
#endif
}

struct BenchResult {
	double build_s;
	double qps;
	double recall;
	double mem_mb;
};

BenchResult RunHnswCore(const std::vector<float> &data, idx_t n, idx_t dim, const std::vector<float> &queries,
                         idx_t nq, idx_t k, const HnswCoreParams &params,
                         const std::vector<std::vector<int64_t>> &truth) {
	DuckDB db(nullptr);
	auto &dbm = DatabaseManager::Get(*db.instance);
	auto attached = dbm.GetDatabase("memory");
	auto &bm = attached->GetStorageManager().GetBlockManager();
	IndexBlockStore store(bm);
	FlatExact quant(dim);
	HnswCore core(params, quant, store);

	const double mem_before = ResidentMB();
	auto t0 = std::chrono::steady_clock::now();
	for (idx_t i = 0; i < n; i++) {
		core.Insert(int64_t(i), data.data() + i * dim);
	}
	auto t1 = std::chrono::steady_clock::now();
	const double build_s = std::chrono::duration<double>(t1 - t0).count();
	const double mem_after = ResidentMB();

	std::vector<float> qp(dim);
	idx_t hits = 0;
	auto s0 = std::chrono::steady_clock::now();
	for (idx_t q = 0; q < nq; q++) {
		quant.PreprocessQuery(queries.data() + q * dim, qp.data());
		auto res = core.Search(qp.data(), k, params.ef_search);
		std::unordered_set<int64_t> truth_set(truth[q].begin(), truth[q].end());
		for (auto &r : res) {
			if (truth_set.count(r.row_id)) {
				hits++;
			}
		}
	}
	auto s1 = std::chrono::steady_clock::now();
	const double search_s = std::chrono::duration<double>(s1 - s0).count();
	return {build_s, double(nq) / search_s, double(hits) / double(nq * k), mem_after - mem_before};
}

} // namespace

int main(int argc, char **argv) {
	idx_t N = 100000;
	idx_t D = 128;
	idx_t NQ = 200;
	idx_t K = 10;
	HnswCoreParams params;
	params.m = 16;
	params.m0 = 32;
	params.ef_construction = 128;
	params.ef_search = 64;
	params.max_level = 8;
	params.seed = 42;

	// Minimal argv: allow overriding N/D for CI-smoke vs. full bench.
	if (argc > 1) {
		N = idx_t(std::atoll(argv[1]));
	}
	if (argc > 2) {
		D = idx_t(std::atoll(argv[2]));
	}
	params.dim = D;

	std::printf("# HnswCore microbench\n");
	std::printf("# N=%lld D=%lld NQ=%lld K=%lld M=%u M0=%u efC=%u efS=%u\n", (long long)N, (long long)D,
	            (long long)NQ, (long long)K, unsigned(params.m), unsigned(params.m0),
	            unsigned(params.ef_construction), unsigned(params.ef_search));

	auto data = MakeGaussian(N, D, 0xCAFE);
	auto queries = MakeGaussian(NQ, D, 0xBEEF);
	std::printf("# brute-forcing ground truth (N=%lld × NQ=%lld × D=%lld)...\n", (long long)N, (long long)NQ,
	            (long long)D);
	auto truth = BruteForce(data.data(), N, D, queries.data(), NQ, K);

	std::printf("# running HnswCore...\n");
	auto hc = RunHnswCore(data, N, D, queries, NQ, K, params, truth);

	std::printf("\n");
	std::printf("engine    build_s   qps       recall@%lld  mem_mb\n", (long long)K);
	std::printf("HnswCore  %7.3f  %7.1f   %7.3f    %6.1f\n", hc.build_s, hc.qps, hc.recall, hc.mem_mb);
	return 0;
}
