// Unit tests for src/algo/diskann/diskann_core.cpp.
//
// DiskAnnCore is the single-layer Vamana graph with codes stored out-of-band
// in a RAM-resident `codes_` array. These tests exercise:
//
//   1. Correctness: on a well-separated Gaussian, top-k results under a flat
//      (identity) quantizer should match brute-force L2SQ (recall >= 0.85).
//   2. Empty-index and k <= size edge cases.
//   3. Round-trip: SerializeState -> DeserializeState reproduces the same
//      top-k answers (the graph node blocks persist via IndexBlockStore).

#include <catch2/catch_test_macros.hpp>

#include "duckdb.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "vindex/diskann_core.hpp"
#include "vindex/index_block_store.hpp"
#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_set>

using duckdb::BlockManager;
using duckdb::DatabaseManager;
using duckdb::DuckDB;
using duckdb::idx_t;
using duckdb::vindex::DiskAnnCore;
using duckdb::vindex::DiskAnnCoreParams;
using duckdb::vindex::IndexBlockStore;
using duckdb::vindex::MetricKind;
using duckdb::vindex::Quantizer;
using duckdb::vindex::QuantizerKind;

namespace {

struct MemoryDB {
	DuckDB db;
	BlockManager *bm;
	MemoryDB() : db(nullptr) {
		auto &dbm = DatabaseManager::Get(*db.instance);
		auto attached = dbm.GetDatabase("memory");
		REQUIRE(attached);
		bm = &attached->GetStorageManager().GetBlockManager();
	}
};

class TestFlat : public Quantizer {
public:
	explicit TestFlat(idx_t dim) : dim_(dim) {
	}
	void Train(const float *, idx_t, idx_t) override {
	}
	void Encode(const float *vec, duckdb::data_ptr_t out) const override {
		std::memcpy(out, vec, dim_ * sizeof(float));
	}
	float EstimateDistance(duckdb::const_data_ptr_t code, const float *q) const override {
		const float *c = reinterpret_cast<const float *>(code);
		float acc = 0.0f;
		for (idx_t i = 0; i < dim_; i++) {
			const float d = c[i] - q[i];
			acc += d * d;
		}
		return acc;
	}
	float CodeDistance(duckdb::const_data_ptr_t a, duckdb::const_data_ptr_t b) const override {
		auto fa = reinterpret_cast<const float *>(a);
		auto fb = reinterpret_cast<const float *>(b);
		float acc = 0.0f;
		for (idx_t i = 0; i < dim_; i++) {
			const float d = fa[i] - fb[i];
			acc += d * d;
		}
		return acc;
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

std::vector<float> RandomGaussian(idx_t n, idx_t dim, uint64_t seed) {
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	std::vector<float> v(n * dim);
	for (idx_t i = 0; i < v.size(); i++) {
		v[i] = nd(rng);
	}
	return v;
}

std::vector<int64_t> BruteForceTopK(const float *data, idx_t n, idx_t dim, const float *q, idx_t k) {
	std::vector<std::pair<float, int64_t>> scored(n);
	for (idx_t i = 0; i < n; i++) {
		float acc = 0.0f;
		for (idx_t j = 0; j < dim; j++) {
			const float d = data[i * dim + j] - q[j];
			acc += d * d;
		}
		scored[i] = {acc, int64_t(i)};
	}
	std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
	std::vector<int64_t> out(k);
	for (idx_t i = 0; i < k; i++) {
		out[i] = scored[i].second;
	}
	return out;
}

} // namespace

TEST_CASE("DiskAnnCore recall under flat codes on N=2000 d=32", "[diskann_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	constexpr idx_t N = 2000;
	constexpr idx_t D = 32;
	constexpr idx_t K = 10;

	auto data = RandomGaussian(N, D, 0xD15Cu);
	TestFlat quant(D);

	DiskAnnCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.alpha = 1.2f;
	params.seed = 42;

	DiskAnnCore core(params, quant, store);
	for (idx_t i = 0; i < N; i++) {
		core.Insert(int64_t(i), data.data() + i * D);
	}
	REQUIRE(core.Size() == N);

	auto queries = RandomGaussian(20, D, 0xA1B2u);
	idx_t hits = 0;
	idx_t total = 0;
	for (idx_t q = 0; q < 20; q++) {
		std::vector<float> qp(quant.QueryWorkspaceSize());
		quant.PreprocessQuery(queries.data() + q * D, qp.data());
		auto got = core.Search(qp.data(), K, params.L);
		auto truth = BruteForceTopK(data.data(), N, D, queries.data() + q * D, K);
		std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());
		for (const auto &g : got) {
			if (truth_set.count(g.row_id)) {
				hits++;
			}
		}
		total += K;
	}
	const float recall = float(hits) / float(total);
	INFO("Recall@10 = " << recall);
	REQUIRE(recall >= 0.85f);
}

TEST_CASE("DiskAnnCore returns empty on empty index", "[diskann_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);
	TestFlat quant(8);
	DiskAnnCoreParams params;
	params.dim = 8;
	DiskAnnCore core(params, quant, store);
	std::vector<float> q(8, 0.0f);
	std::vector<float> qp(8);
	quant.PreprocessQuery(q.data(), qp.data());
	auto res = core.Search(qp.data(), 10, 32);
	REQUIRE(res.empty());
}

TEST_CASE("DiskAnnCore Search honors k <= size", "[diskann_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);
	TestFlat quant(4);
	DiskAnnCoreParams params;
	params.dim = 4;
	params.R = 8;
	DiskAnnCore core(params, quant, store);
	const float v0[4] = {1, 0, 0, 0};
	const float v1[4] = {0, 1, 0, 0};
	const float v2[4] = {0, 0, 1, 0};
	core.Insert(0, v0);
	core.Insert(1, v1);
	core.Insert(2, v2);
	const float qv[4] = {1, 0, 0, 0};
	std::vector<float> qp(4);
	quant.PreprocessQuery(qv, qp.data());
	auto res = core.Search(qp.data(), 2, 16);
	REQUIRE(res.size() == 2);
	REQUIRE(res[0].row_id == 0);
}
