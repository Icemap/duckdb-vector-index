#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "algo/hnsw/hnsw_index.hpp"
#include "algo/hnsw/hnsw_index_scan.hpp"
#include "vindex/vector_index.hpp"
#include "vindex/vector_index_registry.hpp"

namespace duckdb {
namespace vindex {

using hnsw::HnswIndex;
using hnsw::HnswIndexScanBindData;
using hnsw::HnswIndexScanFunction;

// Rewrite AGG(MIN_BY(col1, distance(col2, q), k)) <- SEQ_SCAN(t1)
//     => AGG(MIN_BY(col1, distance(col2, q), k)) <- INDEX_SCAN(t1, q, k × rerank)
//
// The aggregate is preserved intact — the index scan now returns candidates
// (k × rerank_multiple row_ids) and min_by is exactly the exact-distance
// top-k rerank we need. See optimize_scan.cpp for the matching invariant.

class VectorIndexTopKOptimizer : public OptimizerExtension {
public:
	VectorIndexTopKOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		if (plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			return false;
		}
		auto &agg = plan->Cast<LogicalAggregate>();
		if (!agg.groups.empty() || agg.expressions.size() != 1) {
			return false;
		}

		auto &agg_expr = agg.expressions[0];
		if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
			return false;
		}
		auto &agg_func_expr = agg_expr->Cast<BoundAggregateExpression>();
		if (agg_func_expr.function.name != "min_by") {
			return false;
		}
		if (agg_func_expr.children.size() != 3) {
			return false;
		}
		if (agg_func_expr.children[2]->type != ExpressionType::VALUE_CONSTANT) {
			return false;
		}
		const auto &col_expr = agg_func_expr.children[0];
		const auto &dist_expr = agg_func_expr.children[1];
		const auto &limit_expr = agg_func_expr.children[2];

		if (agg.children.size() != 1 || agg.children[0]->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = agg.children[0];
		auto &get = get_ptr->Cast<LogicalGet>();
		if (get.function.name != "seq_scan") {
			return false;
		}
		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			return false;
		}
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		unique_ptr<HnswIndexScanBindData> bind_data;
		vector<reference<Expression>> bindings;

		for (const auto &type_name : VectorIndexRegistry::Instance().TypeNames()) {
			table_info.BindIndexes(context, type_name.c_str());
		}

		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound()) {
				continue;
			}
			auto *vi = VectorIndexRegistry::TryCast(index.Cast<BoundIndex>());
			if (!vi) {
				continue;
			}

			bindings.clear();
			if (!vi->TryMatchDistanceFunction(dist_expr, bindings)) {
				continue;
			}
			unique_ptr<Expression> index_expr;
			if (!vi->TryBindIndexExpression(get, index_expr)) {
				continue;
			}

			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];
			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					continue;
				}
			}

			const auto vector_size = vi->GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}
			const auto k_limit = limit_expr->Cast<BoundConstantExpression>().value.GetValue<int32_t>();
			if (k_limit <= 0 || k_limit >= STANDARD_VECTOR_SIZE) {
				continue;
			}
			auto &hnsw_index = index.Cast<HnswIndex>();
			// Over-fetch: the index returns candidates; the min_by aggregate
			// above picks the exact top-k by real distance.
			const idx_t rerank = vi->GetRerankMultiple(context);
			const idx_t cand_limit = idx_t(k_limit) * rerank;
			bind_data = make_uniq<HnswIndexScanBindData>(duck_table, hnsw_index, cand_limit, std::move(query_vector));
			break;
		}

		if (!bind_data) {
			return false;
		}

		get.function = HnswIndexScanFunction::GetFunction();
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// NOTE: We deliberately do NOT rewrite min_by into list(col ORDER BY dist).
		// The index scan now emits k × rerank_multiple candidates and the
		// original min_by(col, dist, k) is exactly the rerank step we need —
		// picking the k smallest by exact distance. This is the same clean
		// invariant as optimize_scan.cpp: index returns candidates, upstream
		// operator does the exact-distance top-k.

		if (get.table_filters.filters.empty()) {
			return true;
		}

		get.projection_ids.clear();
		get.types.clear();

		auto new_filter = make_uniq<LogicalFilter>();
		auto &column_ids = get.GetColumnIds();
		for (const auto &entry : get.table_filters.filters) {
			idx_t column_id = entry.first;
			auto &type = get.returned_types[column_id];
			bool found = false;
			for (idx_t i = 0; i < column_ids.size(); i++) {
				if (column_ids[i].GetPrimaryIndex() == column_id) {
					column_id = i;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InternalException("Could not find column id for filter");
			}
			auto column = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
			new_filter->expressions.push_back(entry.second->ToExpression(*column));
		}

		new_filter->children.push_back(std::move(get_ptr));
		new_filter->ResolveOperatorTypes();
		get_ptr = std::move(new_filter);
		return true;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		if (!TryOptimize(input.optimizer.binder, input.context, plan)) {
			for (auto &child : plan->children) {
				Optimize(input, child);
			}
		}
	}
};

void RegisterTopKOptimizer(DatabaseInstance &db) {
	ExtensionCallbackManager::Get(db).Register(VectorIndexTopKOptimizer());
}

} // namespace vindex
} // namespace duckdb
