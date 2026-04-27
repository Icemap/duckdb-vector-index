#include "vindex/index_block_store.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/partial_block_manager.hpp"

namespace duckdb {
namespace vindex {

// TODO(m0): full implementation. This file currently provides no-op
// constructors/destructor to satisfy the link step while we port HNSW over.
// Streaming API reuses FixedSizeAllocator + LinkedBlock layout from
// ref/duckdb-vss/src/hnsw/hnsw_index.cpp. Random-access API arrives in M3.

IndexBlockStore::IndexBlockStore(BlockManager &) {
}
IndexBlockStore::~IndexBlockStore() = default;

unique_ptr<IndexBlockStore::StreamWriter> IndexBlockStore::BeginStream(BlockId) {
	throw NotImplementedException("vindex::IndexBlockStore::BeginStream (M0)");
}

unique_ptr<IndexBlockStore::StreamReader> IndexBlockStore::OpenStream(BlockId) const {
	throw NotImplementedException("vindex::IndexBlockStore::OpenStream (M0)");
}

BlockId IndexBlockStore::AllocNode(idx_t) {
	throw NotImplementedException("vindex::IndexBlockStore::AllocNode (M3)");
}
void IndexBlockStore::FreeNode(BlockId) {
	throw NotImplementedException("vindex::IndexBlockStore::FreeNode (M3)");
}
data_ptr_t IndexBlockStore::Pin(BlockId) {
	throw NotImplementedException("vindex::IndexBlockStore::Pin (M3)");
}
void IndexBlockStore::Unpin(BlockId) {
	throw NotImplementedException("vindex::IndexBlockStore::Unpin (M3)");
}

void IndexBlockStore::Init(const IndexStorageInfo &) {
}
IndexStorageInfo IndexBlockStore::GetInfo() const {
	return {};
}
void IndexBlockStore::SerializeBuffers(PartialBlockManager &) {
}
idx_t IndexBlockStore::GetInMemorySize() const {
	return 0;
}
void IndexBlockStore::Reset() {
}

} // namespace vindex
} // namespace duckdb
