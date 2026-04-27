#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
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
//   - Streaming (HNSW, IVF): AppendStream()/ReadStream() write/read a single
//     linear byte stream. Backed internally by a LinkedBlock chain.
//
//   - Random (DiskANN, SPANN): AllocNode()/ReadNode()/Pin()/Unpin() give each
//     node/posting a stable BlockId and let the DuckDB BufferManager evict
//     cold pages. This is the mechanism that lets indexes be larger than RAM.
//
// All storage goes through DuckDB's BlockManager — we never mmap external
// files, satisfying community-extensions requirements.
// ---------------------------------------------------------------------------

using BlockId = IndexPointer;

class IndexBlockStore {
public:
	explicit IndexBlockStore(BlockManager &block_manager);
	~IndexBlockStore();

	// ---- streaming API -----------------------------------------------------
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

	unique_ptr<StreamWriter> BeginStream(BlockId root);
	unique_ptr<StreamReader> OpenStream(BlockId root) const;

	// ---- random-access API (DiskANN/SPANN) ---------------------------------
	BlockId AllocNode(idx_t size);
	void FreeNode(BlockId id);
	data_ptr_t Pin(BlockId id);
	void Unpin(BlockId id);

	// ---- persistence -------------------------------------------------------
	void Init(const IndexStorageInfo &info);
	IndexStorageInfo GetInfo() const;
	void SerializeBuffers(PartialBlockManager &partial_block_manager);
	idx_t GetInMemorySize() const;
	void Reset();

private:
	unique_ptr<FixedSizeAllocator> linked_alloc_; // backs StreamWriter/Reader
	// Node-sized allocator(s) added in M3 for DiskANN/SPANN.
};

} // namespace vindex
} // namespace duckdb
