#include "vindex/index_block_store.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/storage/block.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"

#include <algorithm>
#include <cstring>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// Streaming API: LinkedBlock chain, ported from ref/duckdb-vss/hnsw_index.cpp.
//
// Each segment holds {IndexPointer next_block; char data[...]}; the writer
// fills `data` sequentially and allocates a new segment when it runs out,
// the reader walks the chain. The segment size matches the current HnswIndex
// so a migration from the old LinkedBlock code is a no-op rewrite.
// ---------------------------------------------------------------------------

namespace {

struct StreamBlock {
	static constexpr idx_t SEGMENT_SIZE = Storage::DEFAULT_BLOCK_SIZE - sizeof(validity_t);
	static constexpr idx_t DATA_SIZE = SEGMENT_SIZE - sizeof(IndexPointer);
	static_assert(SEGMENT_SIZE > sizeof(IndexPointer), "Stream segment must hold at least one next-pointer");

	IndexPointer next_block;
	char data[DATA_SIZE];
};

// Stream segments are tagged with STREAM_TAG in the IndexPointer metadata so
// Pin() / AllocForPointer() can route them back to stream_alloc_. We clear
// the tag before handing to FixedSizeAllocator because that API asserts
// metadata bits are zero.
IndexPointer WithTag(IndexPointer raw, NodeSizeId tag) {
	IndexPointer out = raw;
	out.SetMetadata(tag);
	return out;
}

IndexPointer StripTag(IndexPointer tagged) {
	IndexPointer out = tagged;
	out.SetMetadata(0);
	return out;
}

class LinkedBlockWriter final : public IndexBlockStore::StreamWriter {
public:
	LinkedBlockWriter(FixedSizeAllocator &alloc, IndexPointer root)
	    : alloc_(alloc), root_tagged_(root), current_raw_(StripTag(root)), pos_in_block_(0) {
		// Reset the initial block so we always start from a clean slate;
		// avoids leaking stale bytes into the stream on overwrite.
		auto *block = alloc_.Get<StreamBlock>(current_raw_, /*dirty*/ true).get();
		block->next_block.Clear();
		std::memset(block->data, 0, StreamBlock::DATA_SIZE);
	}

	void Write(const_data_ptr_t buffer, idx_t length) override {
		idx_t written = 0;
		while (written < length) {
			auto *block = alloc_.Get<StreamBlock>(current_raw_, /*dirty*/ true).get();
			const idx_t chunk = MinValue<idx_t>(length - written, StreamBlock::DATA_SIZE - pos_in_block_);
			std::memcpy(block->data + pos_in_block_, buffer + written, chunk);
			written += chunk;
			pos_in_block_ += chunk;
			if (pos_in_block_ == StreamBlock::DATA_SIZE) {
				const IndexPointer next = alloc_.New();
				block->next_block = next;
				// Re-fetch the pointer: alloc_.New() may have invalidated the
				// pointer we had (buffer relocation on growth).
				block = alloc_.Get<StreamBlock>(current_raw_, /*dirty*/ true).get();
				block->next_block = next;
				current_raw_ = next;
				pos_in_block_ = 0;
				auto *next_block = alloc_.Get<StreamBlock>(current_raw_, /*dirty*/ true).get();
				next_block->next_block.Clear();
				std::memset(next_block->data, 0, StreamBlock::DATA_SIZE);
			}
		}
	}

	BlockId Root() const override {
		return root_tagged_;
	}

private:
	FixedSizeAllocator &alloc_;
	IndexPointer root_tagged_;
	IndexPointer current_raw_;
	idx_t pos_in_block_;
};

class LinkedBlockReader final : public IndexBlockStore::StreamReader {
public:
	LinkedBlockReader(const FixedSizeAllocator &alloc, IndexPointer root)
	    : alloc_(const_cast<FixedSizeAllocator &>(alloc)), current_raw_(StripTag(root)), pos_in_block_(0) {
	}

	idx_t Read(data_ptr_t buffer, idx_t length) override {
		idx_t read = 0;
		while (read < length) {
			auto *block = alloc_.Get<const StreamBlock>(current_raw_, /*dirty*/ false).get();
			const idx_t chunk = MinValue<idx_t>(length - read, StreamBlock::DATA_SIZE - pos_in_block_);
			std::memcpy(buffer + read, block->data + pos_in_block_, chunk);
			read += chunk;
			pos_in_block_ += chunk;
			if (pos_in_block_ == StreamBlock::DATA_SIZE) {
				current_raw_ = StripTag(block->next_block);
				pos_in_block_ = 0;
				if (current_raw_.Get() == 0) {
					// Ran off the end of the chain.
					return read;
				}
			}
		}
		return read;
	}

private:
	FixedSizeAllocator &alloc_;
	IndexPointer current_raw_;
	idx_t pos_in_block_;
};

} // namespace

// ---------------------------------------------------------------------------
// IndexBlockStore
// ---------------------------------------------------------------------------

IndexBlockStore::IndexBlockStore(BlockManager &block_manager) : block_manager_(block_manager) {
	stream_alloc_ = make_uniq<FixedSizeAllocator>(StreamBlock::SEGMENT_SIZE, block_manager_);
	// Pre-size the per-tag cache to MAX_NODE_SIZES + 1 so we can index by
	// (1-based) node tag directly; stream tag (0xFF) is handled on the slow
	// path since stream pins are not on the HNSW hot path.
	pin_cache_.resize(idx_t(MAX_NODE_SIZES) + 1);
	pin_cache_seg_size_.resize(idx_t(MAX_NODE_SIZES) + 1, 0);
}

IndexBlockStore::~IndexBlockStore() = default;

unique_ptr<IndexBlockStore::StreamWriter> IndexBlockStore::BeginStream(BlockId root) {
	// A valid stream root always carries STREAM_TAG in its metadata byte, so
	// `root.Get() == 0` uniquely identifies the "no existing root" case (the
	// caller passed a default-constructed BlockId). We cannot check
	// `StripTag(root).Get() == 0` because the first allocated IndexPointer is
	// (buffer_id=0, offset=0) whose stripped form is legitimately zero.
	IndexPointer raw_root = (root.Get() == 0) ? stream_alloc_->New() : StripTag(root);
	return make_uniq<LinkedBlockWriter>(*stream_alloc_, WithTag(raw_root, STREAM_TAG));
}

unique_ptr<IndexBlockStore::StreamReader> IndexBlockStore::OpenStream(BlockId root) const {
	if (root.Get() == 0) {
		throw InternalException("IndexBlockStore::OpenStream on null root");
	}
	return make_uniq<LinkedBlockReader>(*stream_alloc_, root);
}

NodeSizeId IndexBlockStore::RegisterNodeSize(idx_t size) {
	if (size == 0) {
		throw InternalException("IndexBlockStore::RegisterNodeSize: size must be positive");
	}
	for (idx_t i = 0; i < node_sizes_.size(); i++) {
		if (node_sizes_[i] == size) {
			// Node size ids are 1-based (tag 0 is reserved for null), see header.
			return static_cast<NodeSizeId>(i + 1);
		}
	}
	if (node_allocs_.size() >= MAX_NODE_SIZES) {
		throw InternalException("IndexBlockStore::RegisterNodeSize: cannot register more than %d sizes",
		                        int(MAX_NODE_SIZES));
	}
	const NodeSizeId id = static_cast<NodeSizeId>(node_allocs_.size() + 1);
	node_allocs_.push_back(make_uniq<FixedSizeAllocator>(size, block_manager_));
	node_sizes_.push_back(size);
	return id;
}

BlockId IndexBlockStore::AllocNode(NodeSizeId size_id) {
	if (size_id == 0 || size_id > node_allocs_.size()) {
		throw InternalException("IndexBlockStore::AllocNode: size_id %d not registered", int(size_id));
	}
	const IndexPointer raw = node_allocs_[size_id - 1]->New();
	return WithTag(raw, size_id);
}

void IndexBlockStore::FreeNode(BlockId id) {
	auto &alloc = AllocForPointer(id);
	alloc.Free(StripTag(id));
}

data_ptr_t IndexBlockStore::Pin(BlockId id) {
	auto &alloc = AllocForPointer(id);
	return alloc.Get(StripTag(id), /*dirty*/ true);
}

void IndexBlockStore::Unpin(BlockId) {
	// FixedSizeAllocator::Get keeps the buffer in memory for us. Once M3
	// enables buffer eviction this needs a proper reference-count decrement,
	// but for M1 this is a no-op.
}

data_ptr_t IndexBlockStore::PinFastSlow(BlockId id) {
	// Cache miss: resolve once via the slow path to get the raw data pointer
	// and segment size, then back-compute the buffer's base so subsequent
	// PinFast calls for any offset in this buffer are pure pointer arithmetic.
	const NodeSizeId tag = id.GetMetadata();
	if (tag == STREAM_TAG || tag == 0 || tag > node_allocs_.size()) {
		// Stream tag and invalid tags fall back to Pin — no cache for them.
		return Pin(id);
	}
	auto &alloc = *node_allocs_[tag - 1];
	const IndexPointer raw = StripTag(id);
	data_ptr_t data_ptr = alloc.Get(raw, /*dirty*/ true);
	const idx_t seg_size = node_sizes_[tag - 1];
	pin_cache_seg_size_[tag] = seg_size;
	data_ptr_t base = data_ptr - raw.GetOffset() * seg_size;
	const idx_t buffer_id = raw.GetBufferId();
	auto &per_tag = pin_cache_[tag];
	if (buffer_id >= per_tag.size()) {
		per_tag.resize(buffer_id + 1, nullptr);
	}
	per_tag[buffer_id] = base;
	return data_ptr;
}

void IndexBlockStore::InvalidatePinCache() {
	for (auto &row : pin_cache_) {
		row.clear();
	}
	std::fill(pin_cache_seg_size_.begin(), pin_cache_seg_size_.end(), 0);
}

FixedSizeAllocator &IndexBlockStore::AllocForPointer(BlockId id) {
	const NodeSizeId tag = id.GetMetadata();
	if (tag == STREAM_TAG) {
		return *stream_alloc_;
	}
	// Tags are 1-based for node allocators; 0 is the null sentinel.
	if (tag == 0 || tag > node_allocs_.size()) {
		throw InternalException("IndexBlockStore: BlockId points to unregistered allocator tag=%d", int(tag));
	}
	return *node_allocs_[tag - 1];
}

const FixedSizeAllocator &IndexBlockStore::AllocForPointer(BlockId id) const {
	const NodeSizeId tag = id.GetMetadata();
	if (tag == STREAM_TAG) {
		return *stream_alloc_;
	}
	if (tag == 0 || tag > node_allocs_.size()) {
		throw InternalException("IndexBlockStore: BlockId points to unregistered allocator tag=%d", int(tag));
	}
	return *node_allocs_[tag - 1];
}

// ---------------------------------------------------------------------------
// Persistence
//
// Layout of IndexStorageInfo.allocator_infos:
//   index 0                    : stream allocator
//   index 1 .. 1 + N           : node allocators in registration order
//
// Init() requires that RegisterNodeSize() has already been called for every
// node allocator the serialized index knows about — the caller (HnswIndex)
// replays registrations from its own options before calling Init().
// ---------------------------------------------------------------------------

void IndexBlockStore::Reset() {
	stream_alloc_->Reset();
	for (auto &alloc : node_allocs_) {
		alloc->Reset();
	}
	InvalidatePinCache();
}

void IndexBlockStore::Init(const IndexStorageInfo &info) {
	if (info.allocator_infos.empty()) {
		// Fresh/empty index — nothing to restore.
		return;
	}
	if (info.allocator_infos.size() != node_allocs_.size() + 1) {
		throw InternalException(
		    "IndexBlockStore::Init: allocator_info count (%llu) != 1 stream + %llu registered node allocators",
		    (unsigned long long)info.allocator_infos.size(), (unsigned long long)node_allocs_.size());
	}
	InvalidatePinCache();
	stream_alloc_->Init(info.allocator_infos[0]);
	for (idx_t i = 0; i < node_allocs_.size(); i++) {
		const auto &ai = info.allocator_infos[i + 1];
		if (ai.segment_size != node_sizes_[i]) {
			throw InternalException(
			    "IndexBlockStore::Init: segment_size mismatch for node allocator %llu (disk=%llu, registered=%llu)",
			    (unsigned long long)i, (unsigned long long)ai.segment_size,
			    (unsigned long long)node_sizes_[i]);
		}
		node_allocs_[i]->Init(ai);
	}
}

IndexStorageInfo IndexBlockStore::GetInfo() const {
	IndexStorageInfo info;
	info.allocator_infos.reserve(1 + node_allocs_.size());
	info.allocator_infos.push_back(stream_alloc_->GetInfo());
	for (const auto &alloc : node_allocs_) {
		info.allocator_infos.push_back(alloc->GetInfo());
	}
	return info;
}

void IndexBlockStore::SerializeBuffers(PartialBlockManager &partial_block_manager) {
	stream_alloc_->SerializeBuffers(partial_block_manager);
	for (auto &alloc : node_allocs_) {
		alloc->SerializeBuffers(partial_block_manager);
	}
	// FixedSizeBuffer::Serialize destroys each buffer_handle, so cached base
	// pointers in pin_cache_ are stale. The next PinFast must re-load through
	// the slow path to re-populate.
	InvalidatePinCache();
}

vector<vector<IndexBufferInfo>> IndexBlockStore::InitSerializationToWAL() {
	vector<vector<IndexBufferInfo>> out;
	out.reserve(1 + node_allocs_.size());
	out.push_back(stream_alloc_->InitSerializationToWAL());
	for (auto &alloc : node_allocs_) {
		out.push_back(alloc->InitSerializationToWAL());
	}
	// WAL serialization also touches the buffers — invalidate for safety even
	// if InitSerializationToWAL doesn't currently destroy handles, so future
	// upstream changes don't silently corrupt our cache.
	InvalidatePinCache();
	return out;
}

idx_t IndexBlockStore::GetInMemorySize() const {
	idx_t total = stream_alloc_->GetInMemorySize();
	for (const auto &alloc : node_allocs_) {
		total += alloc->GetInMemorySize();
	}
	return total;
}


} // namespace vindex
} // namespace duckdb
