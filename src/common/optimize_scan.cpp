#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "vindex/vector_index.hpp"
#include "vindex/vector_index_registry.hpp"
#include "vindex/vector_index_scan.hpp"

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// TopN → index scan rewrite
//
// Registry-driven: iterates every registered VectorIndex type name and asks
// whether it can service the projection's distance expression. The bind data
// and table function are the generic VectorIndexScan{BindData,Function}, so
// adding a new algorithm only requires registering its TYPE_NAME and
// subclassing VectorIndex. No changes here.
// ---------------------------------------------------------------------------

class VectorIndexScanOptimizer : public OptimizerExtension {
public:
	VectorIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		auto &op = *plan;
		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}
		auto &top_n = op.Cast<LogicalTopN>();
		if (top_n.orders.size() != 1) {
			return false;
		}
		const auto &order = top_n.orders[0];
		if (order.type != OrderType::ASCENDING) {
			return false;
		}
		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		if (projection.children.size() != 1 || projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = projection.children.front();
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

		unique_ptr<VectorIndexScanBindData> bind_data;
		vector<reference<Expression>> bindings;

		// Iterate all registered vector-index algorithms.
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
			if (!vi->TryMatchDistanceFunction(projection_expr, bindings)) {
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

			// Over-fetch by `rerank_multiple`. Index returns candidates; the
			// surviving LogicalTopN above does the exact-distance final sort.
			// The scan dispatches through the VectorIndex virtual surface, so
			// we hand it the BoundIndex unchanged.
			const idx_t rerank = vi->GetRerankMultiple(context);
			const idx_t cand_limit = top_n.limit * rerank;
			bind_data =
			    make_uniq<VectorIndexScanBindData>(duck_table, index, cand_limit, std::move(query_vector));
			break;
		}

		if (!bind_data) {
			return false;
		}

		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = VectorIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// NOTE: We intentionally do NOT remove `top_n` from the plan. The index
		// scan now returns candidates (k × rerank_multiple), and the TopN runs
		// the exact-distance ORDER BY + LIMIT over them. This is the clean
		// invariant we want even when rerank_multiple == 1 (TopN over exactly
		// k rows is a no-op but keeps the plan shape uniform).

		// Pull filters up above the index scan (it does not support pushdown).
		get.projection_ids.clear();
		get.types.clear();

		auto new_filter = make_uniq<LogicalFilter>();
		auto &column_ids = get.GetColumnIds();
		for (auto &entry : get.table_filters) {
			idx_t column_id = entry.GetIndex();
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
			auto column =
			    make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, ProjectionIndex(column_id)));
			new_filter->expressions.push_back(entry.Filter().ToExpression(*column));
		}
		new_filter->children.push_back(std::move(get_ptr));
		new_filter->ResolveOperatorTypes();
		get_ptr = std::move(new_filter);

		// TopN stays in the plan — it performs exact-distance rerank + final LIMIT
		// over the (k × rerank_multiple) candidates the index scan emits.
		return true;
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		auto ok = TryOptimize(context, plan);
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];
				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "vindex_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];
						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_scan = OptimizeChildren(input.context, plan);
		if (did_use_scan) {
			MergeProjections(plan);
		}
	}
};

void RegisterScanOptimizer(DatabaseInstance &db) {
	ExtensionCallbackManager::Get(db).Register(VectorIndexScanOptimizer());
}

} // namespace vindex
} // namespace duckdb
