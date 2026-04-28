// Unit tests for src/algo/spann/spann_core.cpp.

#include <catch2/catch_test_macros.hpp>

#include "duckdb.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "vindex/index_block_store.hpp"
#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/spann_core.hpp"

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
using duckdb::vindex::MetricKind;
using duckdb::vindex::Quantizer;
using duckdb::vindex::QuantizerKind;
using duckdb::vindex::SpannCore;
using duckdb::vindex::SpannCoreParams;

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

TEST_CASE("SpannCore recall with replicas beats strict IVF at low nprobe", "[spann_core][unit]") {
	// At nprobe=1 a non-replicated IVF recall plummets because the query's
	// true neighbours are often scattered across multiple Voronoi cells.
	// SPANN's closure replicas are exactly the fix — verify the replicated
	// posting lists give materially better recall than replica_count=1.
	MemoryDB mem;
	const idx_t n = 2000;
	const idx_t dim = 32;
	const idx_t k = 10;
	const idx_t nlist = 128;

	auto data = RandomGaussian(n, dim, /*seed=*/11);

	auto build_and_measure = [&](idx_t replica_count, float closure_factor) {
		IndexBlockStore store(*mem.bm);
		TestFlat q(dim);
		SpannCoreParams params;
		params.nlist = nlist;
		params.nprobe = 1;
		params.dim = dim;
		params.seed = 7;
		params.replica_count = replica_count;
		params.closure_factor = closure_factor;

		SpannCore core(params, q, store);
		core.Train(data.data(), n);
		for (idx_t i = 0; i < n; i++) {
			core.Insert(int64_t(i), data.data() + i * dim);
		}

		const idx_t queries = 20;
		auto q_data = RandomGaussian(queries, dim, /*seed=*/22);
		idx_t hits = 0;
		for (idx_t qi = 0; qi < queries; qi++) {
			const float *qv = q_data.data() + qi * dim;
			auto truth = BruteForceTopK(data.data(), n, dim, qv, k);
			std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());

			auto cands = core.Search(qv, qv, k);
			for (auto &c : cands) {
				if (truth_set.count(c.row_id)) {
					hits++;
				}
			}
		}
		return double(hits) / double(queries * k);
	};

	const double strict_recall = build_and_measure(/*replica_count=*/1, /*closure_factor=*/1.0f);
	const double spann_recall = build_and_measure(/*replica_count=*/8, /*closure_factor=*/1.5f);
	// The whole point of replica writes: at hostile nprobe=1 the query's
	// true neighbours are scattered across cells, and replicating boundary
	// points into multiple cells claws recall back. The *ratio* matters
	// more than the absolute number — strict IVF at nprobe=1 is expected
	// to be poor. Require SPANN to be meaningfully better.
	REQUIRE(spann_recall > strict_recall + 0.10);
}

TEST_CASE("SpannCore Size counts unique rows, Entries counts replica writes", "[spann_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);
	const idx_t n = 200;
	const idx_t dim = 16;
	const idx_t nlist = 16;

	auto data = RandomGaussian(n, dim, /*seed=*/55);
	TestFlat q(dim);
	SpannCoreParams params;
	params.nlist = nlist;
	params.nprobe = 4;
	params.dim = dim;
	params.seed = 42;
	params.replica_count = 4;
	params.closure_factor = 2.0f; // generous — most rows will replicate

	SpannCore core(params, q, store);
	core.Train(data.data(), n);
	for (idx_t i = 0; i < n; i++) {
		core.Insert(int64_t(i), data.data() + i * dim);
	}

	REQUIRE(core.Size() == n);
	REQUIRE(core.Entries() >= core.Size());
	// With replica_count=4 and closure_factor=2.0 on Gaussian data we expect
	// the average replica count to be > 1.0 (otherwise SPANN is a no-op here).
	REQUIRE(core.Entries() > core.Size());
}

TEST_CASE("SpannCore Search dedupes replicated rows", "[spann_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);
	const idx_t n = 200;
	const idx_t dim = 8;
	const idx_t nlist = 16;

	auto data = RandomGaussian(n, dim, /*seed=*/77);
	TestFlat q(dim);
	SpannCoreParams params;
	params.nlist = nlist;
	params.nprobe = nlist; // scan everything — maximises replica collisions
	params.dim = dim;
	params.seed = 1;
	params.replica_count = 4;
	params.closure_factor = 10.0f; // force maximum replication

	SpannCore core(params, q, store);
	core.Train(data.data(), n);
	for (idx_t i = 0; i < n; i++) {
		core.Insert(int64_t(i), data.data() + i * dim);
	}

	auto cands = core.Search(data.data(), data.data(), 10);
	std::unordered_set<int64_t> seen;
	for (auto &c : cands) {
		REQUIRE(seen.insert(c.row_id).second); // no duplicates
	}
}

TEST_CASE("SpannCore serialization round-trips", "[spann_core][unit]") {
	MemoryDB mem;
	IndexBlockStore store_a(*mem.bm);
	IndexBlockStore store_b(*mem.bm);
	const idx_t n = 500;
	const idx_t dim = 16;
	const idx_t nlist = 32;

	auto data = RandomGaussian(n, dim, /*seed=*/33);

	TestFlat q(dim);
	SpannCoreParams params;
	params.nlist = nlist;
	params.nprobe = 8;
	params.dim = dim;
	params.seed = 42;
	params.replica_count = 3;
	params.closure_factor = 1.3f;

	SpannCore a(params, q, store_a);
	a.Train(data.data(), n);
	for (idx_t i = 0; i < n; i++) {
		a.Insert(int64_t(i), data.data() + i * dim);
	}
	duckdb::vector<duckdb::data_t> blob;
	a.SerializeState(blob);

	SpannCore b(params, q, store_b);
	b.DeserializeState(blob.data(), blob.size());
	REQUIRE(b.Size() == a.Size());
	REQUIRE(b.Entries() == a.Entries());

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
