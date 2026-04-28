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

#include "vindex/index_block_store.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/spann_core.hpp"
#include "vindex/vector_index.hpp"

namespace duckdb {
class ColumnDataCollection;
namespace vindex {
namespace spann {

class SpannIndex : public VectorIndex {
public:
	static constexpr const char *TYPE_NAME = "SPANN";

	SpannIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	           TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	           AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	           const IndexStorageInfo &info = IndexStorageInfo(), idx_t estimated_cardinality = 0);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	// --- VectorIndex contract ------------------------------------------------
	MetricKind GetMetricKind() const override;
	idx_t GetVectorSize() const override;
	idx_t GetRerankMultiple(ClientContext &context) const override;

	unique_ptr<IndexScanState> InitializeScan(float *query_vector, idx_t limit, ClientContext &context) override;
	idx_t Scan(IndexScanState &state, Vector &result, idx_t result_offset = 0) override;

	unique_ptr<IndexScanState> InitializeMultiScan(ClientContext &context) override;
	idx_t ExecuteMultiScan(IndexScanState &state, float *query_vector, idx_t limit) override;
	const Vector &GetMultiScanResult(IndexScanState &state) override;
	void ResetMultiScan(IndexScanState &state) override;

	// --- SPANN-specific ------------------------------------------------------
	void Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx);
	void PersistToDisk();
	void Compact() override;

	void TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap = 65536);
	void TrainCentroids(ColumnDataCollection &collection, idx_t sample_cap = 131072);

	void VerifyBuffers(IndexLock &lock) override;

	// --- DuckDB BoundIndex hooks --------------------------------------------
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
		return "Constraint violation in SPANN index";
	}

	void SetDirty() {
		is_dirty = true;
	}
	void SyncSize() {
		index_size = core_ ? core_->Size() : 0;
	}

	idx_t Count() const {
		return core_ ? core_->Size() : 0;
	}
	idx_t NumCentroids() const {
		return params_.nlist;
	}
	idx_t Nprobe() const {
		return params_.nprobe;
	}
	idx_t ReplicaCount() const {
		return params_.replica_count;
	}
	float ClosureFactor() const {
		return params_.closure_factor;
	}
	idx_t Entries() const {
		return core_ ? core_->Entries() : 0;
	}
	idx_t ApproxMemory() const {
		return store_ ? store_->GetInMemorySize() : 0;
	}

private:
	unique_ptr<IndexBlockStore> store_;
	unique_ptr<Quantizer> quantizer_;
	unique_ptr<SpannCore> core_;

	MetricKind metric_ {MetricKind::L2SQ};
	idx_t dim_ = 0;
	SpannCoreParams params_ {};
	idx_t rerank_multiple_ = 1;

	unordered_map<row_t, idx_t> row_to_centroid_;
	unordered_set<row_t> tombstones_;

	BlockId state_root_ {};

	bool is_dirty = false;
	StorageLock rwlock;
	atomic<idx_t> index_size = {0};

	void WriteStateStream();
	void ReadStateStream(BlockId root);
	void InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size, MetricKind metric);
};

} // namespace spann
} // namespace vindex
} // namespace duckdb
