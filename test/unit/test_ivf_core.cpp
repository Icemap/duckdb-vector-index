// Unit tests for src/algo/ivf/ivf_core.cpp.

#include <catch2/catch_test_macros.hpp>

#include "duckdb.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "vindex/index_block_store.hpp"
#include "vindex/ivf_core.hpp"
#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

#include <algorithm>
#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

using duckdb::BlockManager;
using duckdb::DatabaseManager;
using duckdb::DuckDB;
using duckdb::idx_t;
using duckdb::vindex::IndexBlockStore;
using duckdb::vindex::IvfCore;
using duckdb::vindex::IvfCoreParams;
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
	for (auto &x : v) {
		x = nd(rng);
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

TEST_CASE("IvfCore recall under FlatQuantizer on N=2000 d=32", "[ivf_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);
	const idx_t n = 2000;
	const idx_t dim = 32;
	const idx_t k = 10;
	const idx_t nlist = 128;

	auto data = RandomGaussian(n, dim, /*seed=*/11);

	TestFlat q(dim);
	IvfCoreParams params;
	params.nlist = nlist;
	params.nprobe = 32;
	params.dim = dim;
	params.seed = 7;

	IvfCore core(params, q, store);
	core.Train(data.data(), n);
	for (idx_t i = 0; i < n; i++) {
		core.Insert(int64_t(i), data.data() + i * dim);
	}
	REQUIRE(core.Size() == n);

	const idx_t queries = 20;
	auto q_data = RandomGaussian(queries, dim, /*seed=*/22);
	idx_t hits = 0;
	for (idx_t qi = 0; qi < queries; qi++) {
		const float *qv = q_data.data() + qi * dim;
		auto truth = BruteForceTopK(data.data(), n, dim, qv, k);
		std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());

		auto cands = core.Search(qv, qv, k);
		REQUIRE(cands.size() == k);
		for (auto &c : cands) {
			if (truth_set.count(c.row_id)) {
				hits++;
			}
		}
	}
	const double recall = double(hits) / double(queries * k);
	// With nprobe=32/nlist=128 (quarter of the lists) on Gaussian data, expect
	// ≥ 0.85. Loose enough to absorb rng variance but tight enough to catch
	// real regressions.
	REQUIRE(recall >= 0.85);
}

TEST_CASE("IvfCore serialization round-trips", "[ivf_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store_a(*mem.bm);
	IndexBlockStore store_b(*mem.bm);
	const idx_t n = 500;
	const idx_t dim = 16;
	const idx_t nlist = 32;

	auto data = RandomGaussian(n, dim, /*seed=*/33);

	TestFlat q(dim);
	IvfCoreParams params;
	params.nlist = nlist;
	params.nprobe = 8;
	params.dim = dim;
	params.seed = 42;

	IvfCore a(params, q, store_a);
	a.Train(data.data(), n);
	for (idx_t i = 0; i < n; i++) {
		a.Insert(int64_t(i), data.data() + i * dim);
	}
	duckdb::vector<duckdb::data_t> blob;
	a.SerializeState(blob);

	IvfCore b(params, q, store_b);
	b.DeserializeState(blob.data(), blob.size());
	REQUIRE(b.Size() == a.Size());

	// Both produce identical top-k for 10 random queries.
	auto qs = RandomGaussian(10, dim, /*seed=*/44);
	for (idx_t qi = 0; qi < 10; qi++) {
		const float *qv = qs.data() + qi * dim;
		auto ca = a.Search(qv, qv, 5);
		auto cb = b.Search(qv, qv, 5);
		REQUIRE(ca.size() == cb.size());
		for (idx_t i = 0; i < ca.size(); i++) {
			REQUIRE(ca[i].row_id == cb[i].row_id);
		}
	}
}
