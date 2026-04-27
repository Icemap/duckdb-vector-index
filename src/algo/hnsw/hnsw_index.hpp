#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include "vindex/vector_index.hpp"
#include "usearch/duckdb_usearch.hpp"

namespace duckdb {
namespace vindex {
namespace hnsw {

struct HnswIndexStats {
	idx_t max_level;
	idx_t count;
	idx_t capacity;
	idx_t approx_size;
	vector<unum::usearch::index_dense_gt<row_t>::stats_t> level_stats;
};

class HnswIndex : public VectorIndex {
public:
	static constexpr const char *TYPE_NAME = "HNSW";
	using USearchIndexType = unum::usearch::index_dense_gt<row_t>;

	HnswIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	          TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	          AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	          const IndexStorageInfo &info = IndexStorageInfo(), idx_t estimated_cardinality = 0);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	//! The actual usearch index
	USearchIndexType index;

	//! Block pointer to the root of the index
	IndexPointer root_block_ptr;

	//! The allocator used to persist linked blocks
	unique_ptr<FixedSizeAllocator> linked_block_allocator;

	// --- VectorIndex contract (all algorithms) ------------------------------
	MetricKind GetMetricKind() const override;
	idx_t GetVectorSize() const override;

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

	unique_ptr<HnswIndexStats> GetStats();

	void VerifyBuffers(IndexLock &lock) override;

	static const case_insensitive_map_t<unum::usearch::metric_kind_t> METRIC_KIND_MAP;
	static const unordered_map<uint8_t, unum::usearch::scalar_kind_t> SCALAR_KIND_MAP;

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
		index_size = index.size();
	}

private:
	bool is_dirty = false;
	StorageLock rwlock;
	atomic<idx_t> index_size = {0};
};

} // namespace hnsw
} // namespace vindex
} // namespace duckdb
