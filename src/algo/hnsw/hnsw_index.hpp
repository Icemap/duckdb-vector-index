#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include "vindex/hnsw_core.hpp"
#include "vindex/index_block_store.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/vector_index.hpp"

namespace duckdb {
class ColumnDataCollection;
namespace vindex {
namespace hnsw {

struct HnswLevelStats {
	idx_t nodes;
	idx_t edges;
	idx_t max_edges;
	idx_t allocated_bytes;
};

struct HnswIndexStats {
	idx_t max_level;
	idx_t count;
	idx_t capacity;
	idx_t approx_size;
	vector<HnswLevelStats> level_stats;
};

class HnswIndex : public VectorIndex {
public:
	static constexpr const char *TYPE_NAME = "HNSW";

	HnswIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	          TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	          AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	          const IndexStorageInfo &info = IndexStorageInfo(), idx_t estimated_cardinality = 0);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	// --- VectorIndex contract (all algorithms) ------------------------------
	MetricKind GetMetricKind() const override;
	idx_t GetVectorSize() const override;
	idx_t GetRerankMultiple(ClientContext &context) const override;

	unique_ptr<IndexScanState> InitializeScan(float *query_vector, idx_t limit, ClientContext &context) override;
	idx_t Scan(IndexScanState &state, Vector &result, idx_t result_offset = 0) override;

	unique_ptr<IndexScanState> InitializeMultiScan(ClientContext &context) override;
	idx_t ExecuteMultiScan(IndexScanState &state, float *query_vector, idx_t limit) override;
	const Vector &GetMultiScanResult(IndexScanState &state) override;
	void ResetMultiScan(IndexScanState &state) override;

	// --- HNSW-specific ------------------------------------------------------
	void Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx);
	void PersistToDisk();
	void Compact();

	// One-shot training pass over up to `sample_cap` input vectors. Called by
	// PhysicalCreateHnswIndex::Finalize before the parallel Construct phase.
	// FlatQuantizer's Train is a no-op; RabitqQuantizer derives centroid +
	// rotation seed from this sample. Safe to call exactly once per index.
	void TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap = 65536);

	unique_ptr<HnswIndexStats> GetStats();

	void VerifyBuffers(IndexLock &lock) override;

	// --- DuckDB BoundIndex hooks -------------------------------------------
	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	void CommitDrop(IndexLock &index_lock) override;
	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;

	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;
	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;

	idx_t GetInMemorySize(IndexLock &state) override;
	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
	void Vacuum(IndexLock &state) override;
	void Verify(IndexLock &state) override;
	string ToString(IndexLock &state, bool display_ascii = false) override;
	void VerifyAllocations(IndexLock &state) override;

	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override {
		return "Constraint violation in HNSW index";
	}

	void SetDirty() {
		is_dirty = true;
	}
	void SyncSize() {
		index_size = core_ ? core_->Size() : 0;
	}

private:
	// Core algorithm + storage. Created in the constructor (from WITH options or
	// from persisted state). Construct() / Scan() delegate here.
	unique_ptr<IndexBlockStore> store_;
	unique_ptr<Quantizer> quantizer_;
	unique_ptr<HnswCore> core_;

	// Hyperparameters snapshot — kept separately so GetVectorSize/GetMetricKind
	// work before core_ is lazily resized, and so stats can report them.
	MetricKind metric_ {MetricKind::L2SQ};
	idx_t dim_ = 0;
	HnswCoreParams params_ {};
	// Index-level default for the rerank over-fetch factor. 1 = no over-fetch
	// (pre-M1.6e behavior). Session pragma `vindex_rerank_multiple` overrides.
	idx_t rerank_multiple_ = 1;

	// row_t → BlockId map: required to honor DuckDB's Delete(row_ids) contract
	// since HnswCore addresses nodes by BlockId, not row_id. usearch had this
	// built in via its internal key hash; we track it at this layer.
	unordered_map<row_t, BlockId> row_to_block_;
	// Tombstones — HnswCore's graph retains the node (edges might still point
	// at it) but Scan filters the row_id out. Compact() rebuilds the graph.
	unordered_set<row_t> tombstones_;

	// Root of the HnswIndex state stream (quantizer blob + HnswCore state +
	// row_to_block_ + tombstones_). The node allocators persist separately via
	// IndexBlockStore::GetInfo.
	BlockId state_root_ {};

	bool is_dirty = false;
	StorageLock rwlock;
	atomic<idx_t> index_size = {0};

	// Serialize HnswIndex-level state into the state stream. The caller has
	// typically just written node allocators into the block store.
	void WriteStateStream();
	// Load quantizer / core / row map from an existing state stream root.
	void ReadStateStream(BlockId root);
	// Build a fresh core_ + quantizer_ from WITH-options for a new index.
	void InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size,
	                     MetricKind metric);
};

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
