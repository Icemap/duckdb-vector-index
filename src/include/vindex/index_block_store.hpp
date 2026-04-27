#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/index_storage_info.hpp"

namespace duckdb {
class BlockManager;
class PartialBlockManager;
} // namespace duckdb

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// IndexBlockStore — addressable block storage owned by a VectorIndex.
//
// Replaces the ad-hoc LinkedBlock writer used by ref/duckdb-vss/hnsw_index.cpp.
// Designed to satisfy two very different access patterns:
//
//   - Streaming (HNSW save_to_stream, IVF postings): BeginStream()/OpenStream()
//     write/read a single linear byte stream backed by a LinkedBlock chain.
//
//   - Random (HnswCore graph nodes, DiskANN, SPANN): RegisterNodeSize()
//     returns a NodeSizeId; AllocNode(id)/FreeNode/Pin/Unpin give each node a
//     stable BlockId and let the DuckDB BufferManager evict cold pages. This
//     is the mechanism that lets indexes be larger than RAM.
//
// All storage goes through DuckDB's BlockManager — we never mmap external
// files, satisfying community-extensions requirements.
//
// IndexPointer layout (8 bytes) — reused from DuckDB's FixedSizeAllocator:
//   [63:56] metadata — used here to tag which allocator the block belongs to:
//                       0        = null / unused (the default BlockId)
//                       1..N     = registered node sizes, in registration order
//                       STREAM_TAG (0xFF) = LinkedBlock stream segments
//   [55:32] offset   — segment index inside the buffer
//   [31:0]  buffer_id
//
// Reserving tag 0 as "null" is what lets a default-constructed BlockId serve
// as a sentinel: the very first allocation (buffer_id=0, offset=0) would
// otherwise collide with Get()==0 if tag 0 were a valid node tag.
// ---------------------------------------------------------------------------

using BlockId = IndexPointer;
using NodeSizeId = uint8_t;

class IndexBlockStore {
public:
	// Sentinel metadata byte for IndexPointers that belong to the streaming
	// allocator. Node allocators use 1..MAX_NODE_SIZES; tag 0 is reserved as
	// the null / default-constructed BlockId.
	static constexpr NodeSizeId STREAM_TAG = 0xFF;
	// Practical cap on how many distinct node sizes an index can register.
	// HnswCore registers one per level (max_level + 1 ≤ 16), DiskANN + SPANN
	// need 2-3. 16 leaves tag 0xFF free for the streaming sentinel.
	static constexpr NodeSizeId MAX_NODE_SIZES = 16;

public:
	explicit IndexBlockStore(BlockManager &block_manager);
	~IndexBlockStore();

	// ---- streaming API -----------------------------------------------------
	//
	// StreamWriter writes a contiguous byte stream into a chain of LinkedBlock
	// segments. `Root()` returns the head of that chain; persist it and pass
	// to OpenStream later to read back the same bytes.
	class StreamWriter {
	public:
		virtual ~StreamWriter() = default;
		virtual void Write(const_data_ptr_t buffer, idx_t length) = 0;
		virtual BlockId Root() const = 0;
	};

	class StreamReader {
	public:
		virtual ~StreamReader() = default;
		virtual idx_t Read(data_ptr_t buffer, idx_t length) = 0;
	};

	// If `root` is the null BlockId, allocate a new chain and start at the
	// beginning. If `root` is non-null, *overwrite* the chain starting at
	// that block (reusing already-allocated blocks, extending if needed).
	unique_ptr<StreamWriter> BeginStream(BlockId root);
	unique_ptr<StreamReader> OpenStream(BlockId root) const;

	// ---- random-access API -------------------------------------------------
	//
	// Register once at index-construction time; holds a dedicated
	// FixedSizeAllocator. Returns a NodeSizeId in [0, MAX_NODE_SIZES) that
	// must be passed back to AllocNode. Re-registering the same size returns
	// the same id (idempotent).
	NodeSizeId RegisterNodeSize(idx_t size);

	BlockId AllocNode(NodeSizeId size_id);
	void FreeNode(BlockId id);
	// Pin returns a pointer valid as long as the caller holds the store
	// reference. In the M1 implementation the underlying FixedSizeAllocator
	// keeps every buffer in memory, so Unpin is a no-op. When M3 enables disk
	// eviction, these become proper pin/unpin reference counts.
	data_ptr_t Pin(BlockId id);
	void Unpin(BlockId id);

	// Hot-path fast pin. Bypasses `FixedSizeAllocator::Get`, which on every call
	// does an `unordered_map<idx_t, unique_ptr<FixedSizeBuffer>>::find()` plus a
	// `std::mutex` acquisition inside `FixedSizeBuffer::GetDeprecated()`. HNSW
	// search calls Pin ~`ef_search` times per query, so that per-call overhead
	// dominates the traversal cost vs usearch (which resolves nodes via pointer
	// arithmetic on a single arena).
	//
	// We cache the base pointer of every FixedSizeBuffer we have touched,
	// keyed by (tag, buffer_id). A cache hit is a flat-vector lookup + pointer
	// arithmetic — no hash, no mutex. Safe under M1's "buffers are never
	// unloaded once touched" invariant; must be invalidated on Init()/Reset()
	// (calls below).
	data_ptr_t PinFast(BlockId id) {
		const NodeSizeId tag = id.GetMetadata();
		const idx_t buffer_id = id.GetBufferId();
		auto &per_tag = pin_cache_[tag];
		if (buffer_id < per_tag.size() && per_tag[buffer_id] != nullptr) {
			return per_tag[buffer_id] + id.GetOffset() * pin_cache_seg_size_[tag];
		}
		return PinFastSlow(id);
	}

	// ---- persistence -------------------------------------------------------
	// `info.allocator_infos` is laid out as:
	//   [0]                          : stream allocator
	//   [1 .. 1+node_allocs_.size())  : registered node allocators, in id order
	void Init(const IndexStorageInfo &info);
	IndexStorageInfo GetInfo() const;
	void SerializeBuffers(PartialBlockManager &partial_block_manager);
	// WAL path: returns per-allocator buffer lists, outer index matches the
	// allocator_infos layout above.
	vector<vector<IndexBufferInfo>> InitSerializationToWAL();
	idx_t GetInMemorySize() const;
	void Reset();

	// Number of node allocators currently registered. Exposed for tests and
	// for the HnswIndex storage-info round-trip.
	idx_t NodeAllocCount() const {
		return node_allocs_.size();
	}

private:
	BlockManager &block_manager_;
	unique_ptr<FixedSizeAllocator> stream_alloc_;
	vector<unique_ptr<FixedSizeAllocator>> node_allocs_;
	vector<idx_t> node_sizes_; // parallel to node_allocs_

	// PinFast cache — indexed by [tag][buffer_id] → base pointer of that
	// buffer, plus per-tag segment size for offset math. Sized to MAX_NODE_SIZES
	// + 1 so STREAM_TAG (0xFF) is not pre-allocated; stream pins don't use it.
	mutable vector<vector<data_ptr_t>> pin_cache_;
	mutable vector<idx_t> pin_cache_seg_size_;
	data_ptr_t PinFastSlow(BlockId id);
	void InvalidatePinCache();

	FixedSizeAllocator &AllocForPointer(BlockId id);
	const FixedSizeAllocator &AllocForPointer(BlockId id) const;
};

} // namespace vindex
} // namespace duckdb
