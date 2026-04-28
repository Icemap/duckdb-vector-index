#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include "vindex/metric.hpp"

namespace duckdb {
class LogicalGet;
} // namespace duckdb

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// VectorIndex — the common DuckDB-side facade for every ANN algorithm in this
// extension. Concrete algorithms (HNSW, IVF, DiskANN, ...) subclass and
// override the virtual hooks.
//
// Why this exists: ref/duckdb-vss hard-codes HNSWIndex::TYPE_NAME inside four
// optimizer extensions; each new algorithm would require duplicating those.
// All optimizers in src/common/ work against VectorIndex*, so adding an
// algorithm only requires registering a new subclass via
// VectorIndexRegistry::Register().
//
// The virtual surface mirrors what the existing HNSW code paths need:
//   - single-query scan (optimize_scan / optimize_topk)
//   - multi-query scan   (optimize_join)
//   - metric introspection (optimize_expr, pragma info)
//   - distance-function matching against the column expression
// ---------------------------------------------------------------------------

class VectorIndex : public BoundIndex {
public:
	VectorIndex(const string &name, const string &type_name, IndexConstraintType constraint_type,
	            const vector<column_t> &column_ids, TableIOManager &table_io_manager,
	            const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db);

	// --- introspection -------------------------------------------------------
	virtual MetricKind GetMetricKind() const = 0;
	virtual idx_t GetVectorSize() const = 0;
	string GetMetric() const {
		return MetricName(GetMetricKind());
	}

	// Over-fetch factor used by the optimizers when asking the index for
	// candidates. The index returns `k × GetRerankMultiple()` row_ids; the
	// upstream TopN / min_by aggregate computes exact distance and keeps the
	// final top-k. Session override (`vindex_rerank_multiple`) takes precedence
	// over the index-level default.
	virtual idx_t GetRerankMultiple(ClientContext &context) const = 0;

	// Reclaims tombstoned entries. Called from `PRAGMA vindex_compact_index`.
	// Each algorithm decides what "compact" means (IVF rewrites posting lists
	// in place, HNSW/DiskANN currently drop the tombstone set and mark the
	// index dirty — full graph rebuild still requires CREATE INDEX).
	virtual void Compact() = 0;

	// --- expression matching (used by every optimizer) ----------------------
	// Returns true if `expr` is a distance function call whose arguments
	// include this index's column and a constant vector of the right shape.
	bool TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
	                              vector<reference<Expression>> &bindings) const;

	// Rewrites the unbound index expression against the column layout in `get`.
	bool TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const;

	// --- single-query scan (scan / topk optimizers) --------------------------
	// Returns a state that yields up to `limit` approximate-nearest-neighbour
	// row_ids via repeated Scan() calls.
	virtual unique_ptr<IndexScanState> InitializeScan(float *query_vector, idx_t limit, ClientContext &context) = 0;
	virtual idx_t Scan(IndexScanState &state, Vector &result, idx_t result_offset = 0) = 0;

	// --- multi-query scan (join optimizer) -----------------------------------
	virtual unique_ptr<IndexScanState> InitializeMultiScan(ClientContext &context) = 0;
	virtual idx_t ExecuteMultiScan(IndexScanState &state, float *query_vector, idx_t limit) = 0;
	virtual const Vector &GetMultiScanResult(IndexScanState &state) = 0;
	virtual void ResetMultiScan(IndexScanState &state) = 0;

private:
	// Built lazily on first TryMatchDistanceFunction() call, when the subclass
	// has finished constructing and GetMetricKind()/GetVectorSize() are safe.
	mutable unique_ptr<ExpressionMatcher> function_matcher_;
	unique_ptr<ExpressionMatcher> BuildDistanceMatcher() const;
};

} // namespace vindex
} // namespace duckdb
