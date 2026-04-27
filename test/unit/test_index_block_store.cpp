// Unit tests for src/common/index_block_store.cpp.
//
// IndexBlockStore is the foundation layer for every algorithm we ship (HNSW,
// IVF, DiskANN, SPANN): one test file covering both the streaming chain and
// the random-access node allocators buys us a lot of safety.
//
// Each test owns its own in-memory DuckDB instance, pulls out the
// BlockManager, and drives IndexBlockStore directly; nothing here touches SQL
// or the extension surface.

#include <catch2/catch_test_macros.hpp>

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "vindex/index_block_store.hpp"

#include <cstdint>
#include <cstring>
#include <random>

using duckdb::BlockManager;
using duckdb::DatabaseManager;
using duckdb::DuckDB;
using duckdb::IndexStorageInfo;
using duckdb::Storage;
using duckdb::const_data_ptr_t;
using duckdb::data_ptr_t;
using duckdb::data_t;
using duckdb::idx_t;

using duckdb::vindex::BlockId;
using duckdb::vindex::IndexBlockStore;
using duckdb::vindex::NodeSizeId;

namespace {

// Boilerplate: an in-memory DuckDB gives us a real BlockManager without
// spinning up a file on disk.
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

} // namespace

TEST_CASE("IndexBlockStore streaming round-trip for a single-block payload",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	// ~1 KiB of payload fits inside a single LinkedBlock segment.
	constexpr idx_t N = 1024;
	std::vector<uint8_t> src(N);
	for (idx_t i = 0; i < N; i++) {
		src[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
	}

	BlockId root;
	{
		auto writer = store.BeginStream(BlockId());
		writer->Write(reinterpret_cast<const_data_ptr_t>(src.data()), N);
		root = writer->Root();
	}
	REQUIRE(root.Get() != 0);

	std::vector<uint8_t> dst(N, 0);
	{
		auto reader = store.OpenStream(root);
		const idx_t n = reader->Read(reinterpret_cast<data_ptr_t>(dst.data()), N);
		REQUIRE(n == N);
	}
	REQUIRE(std::memcmp(src.data(), dst.data(), N) == 0);
}

TEST_CASE("IndexBlockStore streaming round-trip across many block boundaries",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	// Pick a payload that forces the writer to chain multiple segments.
	const idx_t segment_data =
	    Storage::DEFAULT_BLOCK_SIZE - sizeof(duckdb::validity_t) - sizeof(duckdb::IndexPointer);
	const idx_t N = segment_data * 3 + 12345; // 3 full segments + a tail

	std::mt19937_64 rng(0xC0DE);
	std::vector<uint8_t> src(N);
	for (idx_t i = 0; i < N; i++) {
		src[i] = static_cast<uint8_t>(rng());
	}

	BlockId root;
	{
		auto writer = store.BeginStream(BlockId());
		// Write in uneven chunks to exercise the straddle path.
		idx_t off = 0;
		const idx_t chunks[] = {7, 4096, 1, segment_data - 3, 500, N};
		for (idx_t target : chunks) {
			if (off >= N) {
				break;
			}
			const idx_t want = duckdb::MinValue<idx_t>(target, N - off);
			writer->Write(reinterpret_cast<const_data_ptr_t>(src.data() + off), want);
			off += want;
		}
		REQUIRE(off == N);
		root = writer->Root();
	}

	std::vector<uint8_t> dst(N, 0);
	{
		auto reader = store.OpenStream(root);
		const idx_t n = reader->Read(reinterpret_cast<data_ptr_t>(dst.data()), N);
		REQUIRE(n == N);
	}
	REQUIRE(std::memcmp(src.data(), dst.data(), N) == 0);
}

TEST_CASE("IndexBlockStore BeginStream with an existing root overwrites the chain",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	constexpr idx_t N = 4096;
	std::vector<uint8_t> first(N, 0xAA);
	std::vector<uint8_t> second(N, 0x55);

	BlockId root;
	{
		auto writer = store.BeginStream(BlockId());
		writer->Write(reinterpret_cast<const_data_ptr_t>(first.data()), N);
		root = writer->Root();
	}
	{
		auto writer = store.BeginStream(root);
		writer->Write(reinterpret_cast<const_data_ptr_t>(second.data()), N);
		REQUIRE(writer->Root().Get() == root.Get());
	}
	std::vector<uint8_t> dst(N, 0);
	{
		auto reader = store.OpenStream(root);
		REQUIRE(reader->Read(reinterpret_cast<data_ptr_t>(dst.data()), N) == N);
	}
	REQUIRE(std::memcmp(second.data(), dst.data(), N) == 0);
}

TEST_CASE("IndexBlockStore RegisterNodeSize is idempotent and capped",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	const NodeSizeId a = store.RegisterNodeSize(64);
	const NodeSizeId b = store.RegisterNodeSize(64);
	REQUIRE(a == b);
	REQUIRE(store.NodeAllocCount() == 1);

	const NodeSizeId c = store.RegisterNodeSize(128);
	REQUIRE(c != a);
	REQUIRE(store.NodeAllocCount() == 2);

	// Registering more than MAX_NODE_SIZES must throw.
	// We already hold 2 slots (64, 128); fill the remaining 6 with distinct
	// sizes, then expect the next call to throw.
	for (idx_t i = 0; i < IndexBlockStore::MAX_NODE_SIZES - 2; i++) {
		store.RegisterNodeSize(256 + i);
	}
	REQUIRE(store.NodeAllocCount() == IndexBlockStore::MAX_NODE_SIZES);
	REQUIRE_THROWS_AS(store.RegisterNodeSize(9999), duckdb::InternalException);

	REQUIRE_THROWS_AS(store.RegisterNodeSize(0), duckdb::InternalException);
}

TEST_CASE("IndexBlockStore AllocNode / Pin / FreeNode round-trip preserves bytes",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	const NodeSizeId small = store.RegisterNodeSize(48);
	const NodeSizeId big = store.RegisterNodeSize(200);

	// Allocate a handful of nodes at each size, scribble a tagged payload,
	// then read back. This also exercises that Pin() correctly routes back
	// through the metadata tag.
	std::vector<std::pair<BlockId, uint8_t>> handles;
	for (int i = 0; i < 16; i++) {
		const NodeSizeId sz = (i % 2 == 0) ? small : big;
		BlockId id = store.AllocNode(sz);
		const uint8_t tag = static_cast<uint8_t>(i + 1);
		data_ptr_t mem_ptr = store.Pin(id);
		const idx_t bytes = (sz == small) ? 48 : 200;
		std::memset(mem_ptr, tag, bytes);
		store.Unpin(id);
		handles.emplace_back(id, tag);
	}
	for (auto &h : handles) {
		data_ptr_t mem_ptr = store.Pin(h.first);
		const idx_t bytes = (h.first.GetMetadata() == small) ? 48 : 200;
		for (idx_t k = 0; k < bytes; k++) {
			REQUIRE(mem_ptr[k] == h.second);
		}
		store.Unpin(h.first);
	}

	// Freeing should not affect the still-live handles.
	const BlockId victim = handles.back().first;
	handles.pop_back();
	store.FreeNode(victim);
	for (auto &h : handles) {
		data_ptr_t mem_ptr = store.Pin(h.first);
		const idx_t bytes = (h.first.GetMetadata() == small) ? 48 : 200;
		for (idx_t k = 0; k < bytes; k++) {
			REQUIRE(mem_ptr[k] == h.second);
		}
	}
}

TEST_CASE("IndexBlockStore GetInfo layout matches stream + nodes in registration order",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	const NodeSizeId s64 = store.RegisterNodeSize(64);
	const NodeSizeId s128 = store.RegisterNodeSize(128);

	// Allocate some state so the allocators are non-empty.
	{
		auto w = store.BeginStream(BlockId());
		const std::vector<uint8_t> payload(5000, 0x42);
		w->Write(reinterpret_cast<const_data_ptr_t>(payload.data()), payload.size());
	}
	for (int i = 0; i < 4; i++) {
		store.AllocNode(s64);
		store.AllocNode(s128);
	}

	IndexStorageInfo info = store.GetInfo();
	REQUIRE(info.allocator_infos.size() == 3);
	REQUIRE(info.allocator_infos[1].segment_size == 64);
	REQUIRE(info.allocator_infos[2].segment_size == 128);
}

TEST_CASE("IndexBlockStore Init rejects mismatched segment sizes",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	store.RegisterNodeSize(64);

	IndexStorageInfo info = store.GetInfo();
	// Simulate a registration mismatch: change the second allocator's segment_size.
	info.allocator_infos[1].segment_size = 999;

	IndexBlockStore other(*mem.bm);
	other.RegisterNodeSize(64);
	REQUIRE_THROWS_AS(other.Init(info), duckdb::InternalException);
}

TEST_CASE("IndexBlockStore Init with the wrong allocator count throws",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);
	store.RegisterNodeSize(64);

	// Build a bad info with 3 allocator_infos but only 1 node size registered on the target.
	IndexStorageInfo info = store.GetInfo();
	info.allocator_infos.push_back(info.allocator_infos[1]);

	IndexBlockStore target(*mem.bm);
	target.RegisterNodeSize(64);
	REQUIRE_THROWS_AS(target.Init(info), duckdb::InternalException);
}

TEST_CASE("IndexBlockStore Reset empties the allocators",
          "[index_block_store][unit]") {
	MemoryDB mem;
	IndexBlockStore store(*mem.bm);

	const NodeSizeId s64 = store.RegisterNodeSize(64);
	for (int i = 0; i < 8; i++) {
		store.AllocNode(s64);
	}
	REQUIRE(store.GetInMemorySize() > 0);

	store.Reset();
	// After Reset we should still be able to allocate again. The tag byte is
	// non-zero (node tags are 1-based), so id.Get() must be non-zero.
	BlockId id = store.AllocNode(s64);
	REQUIRE(id.Get() != 0);
}
