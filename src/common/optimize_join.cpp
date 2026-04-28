#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"

#include "algo/hnsw/hnsw_index.hpp"
#include "vindex/vector_index.hpp"
#include "vindex/vector_index_registry.hpp"

namespace duckdb {
namespace vindex {

using hnsw::HnswIndex;

//------------------------------------------------------------------------------
// Physical Operator — delegates to the VectorIndex multi-scan API.
// Works against any VectorIndex, but for M0 we only ever bind HnswIndex.
//------------------------------------------------------------------------------

class PhysicalVectorIndexJoin final : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalVectorIndexJoin(PhysicalPlan &physical_plan, const vector<LogicalType> &types_p,
	                        const idx_t estimated_cardinality, DuckTableEntry &table_p, VectorIndex &index_p,
	                        const idx_t limit_p)
	    : PhysicalOperator(physical_plan, TYPE, types_p, estimated_cardinality), table(table_p), index(index_p),
	      limit(limit_p) {
	}

	string GetName() const override {
		return "VECTOR_INDEX_JOIN";
	}
	bool ParallelOperator() const override {
		return false;
	}
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	DuckTableEntry &table;
	VectorIndex &index;
	idx_t limit;

	vector<column_t> inner_column_ids;
	vector<idx_t> inner_projection_ids;

	idx_t outer_vector_column;
	idx_t inner_vector_column;
};

class VectorIndexJoinState final : public OperatorState {
public:
	idx_t input_idx = 0;

	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<StorageIndex> physical_column_ids;

	unique_ptr<IndexScanState> index_state;
	SelectionVector match_sel;
};

unique_ptr<OperatorState> PhysicalVectorIndexJoin::GetOperatorState(ExecutionContext &context) const {
	auto result = make_uniq<VectorIndexJoinState>();

	auto &local_storage = LocalStorage::Get(context.client, table.catalog);
	result->physical_column_ids.reserve(inner_column_ids.size());
	for (auto &id : inner_column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->physical_column_ids.emplace_back(col_id);
	}

	result->match_sel.Initialize();
	result->local_storage_state.Initialize(result->physical_column_ids, nullptr);
	local_storage.InitializeScan(table.GetStorage(), result->local_storage_state.local_state, nullptr);

	result->index_state = index.InitializeMultiScan(context.client);
	return std::move(result);
}

OperatorResultType PhysicalVectorIndexJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate, OperatorState &ostate) const {
	auto &state = ostate.Cast<VectorIndexJoinState>();
	auto &transaction = DuckTransaction::Get(context.client, table.catalog);

	input.Flatten();

	const auto MATCH_COLUMN_OFFSET = inner_column_ids.size();
	const auto OUTER_COLUMN_OFFSET = MATCH_COLUMN_OFFSET + 1;

	auto &rhs_vector_vector = input.data[outer_vector_column];
	auto &rhs_vector_child = ArrayVector::GetEntry(rhs_vector_vector);
	const auto rhs_vector_size = ArrayType::GetSize(rhs_vector_vector.GetType());
	const auto rhs_vector_ptr = FlatVector::GetData<float>(rhs_vector_child);

	const auto row_number_vector = FlatVector::GetData<int64_t>(chunk.data[MATCH_COLUMN_OFFSET]);

	index.ResetMultiScan(*state.index_state);

	const auto batch_count = MinValue(input.size() - state.input_idx, STANDARD_VECTOR_SIZE / limit);
	idx_t output_idx = 0;
	for (idx_t batch_idx = 0; batch_idx < batch_count; batch_idx++, state.input_idx++) {
		const auto rhs_vector_data = rhs_vector_ptr + state.input_idx * rhs_vector_size;

		const auto match_count = index.ExecuteMultiScan(*state.index_state, rhs_vector_data, limit);
		for (idx_t i = 0; i < match_count; i++) {
			state.match_sel.set_index(output_idx, state.input_idx);
			row_number_vector[output_idx] = i + 1;
			output_idx++;
		}
	}

	const auto &row_ids = index.GetMultiScanResult(*state.index_state);

	table.GetStorage().Fetch(transaction, chunk, state.physical_column_ids, row_ids, output_idx, state.fetch_state);
	chunk.Slice(input, state.match_sel, output_idx, OUTER_COLUMN_OFFSET);
	chunk.SetCardinality(output_idx);

	if (state.input_idx == input.size()) {
		state.input_idx = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalVectorIndexJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result.insert("table", table.name);
	result.insert("index", index.GetIndexName());
	result.insert("limit", to_string(limit));
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

//------------------------------------------------------------------------------
// Logical Operator
//------------------------------------------------------------------------------

class LogicalVectorIndexJoin final : public LogicalExtensionOperator {
public:
	LogicalVectorIndexJoin(const idx_t table_index_p, DuckTableEntry &table_p, VectorIndex &index_p,
	                       const idx_t limit_p)
	    : table_index(table_index_p), table(table_p), index(index_p), limit(limit_p) {
	}

	string GetName() const override {
		return "VECTOR_INDEX_JOIN";
	}
	void ResolveTypes() override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<ColumnBinding> GetLeftBindings();
	vector<ColumnBinding> GetRightBindings();
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	idx_t EstimateCardinality(ClientContext &context) override;

	idx_t table_index;
	DuckTableEntry &table;
	VectorIndex &index;
	idx_t limit;

	vector<column_t> inner_column_ids;
	vector<idx_t> inner_projection_ids;
	vector<LogicalType> inner_returned_types;

	idx_t outer_vector_column;
	idx_t inner_vector_column;
};

void LogicalVectorIndexJoin::ResolveTypes() {
	if (inner_column_ids.empty()) {
		inner_column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}
	types.clear();

	if (inner_projection_ids.empty()) {
		for (const auto &i : inner_column_ids) {
			if (i == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(inner_returned_types[i]);
			}
		}
	} else {
		for (const auto &proj_index : inner_projection_ids) {
			const auto &i = inner_column_ids[proj_index];
			if (i == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(inner_returned_types[i]);
			}
		}
	}

	types.emplace_back(LogicalType::BIGINT);
	auto &right_types = children[0]->types;
	types.insert(types.end(), right_types.begin(), right_types.end());
}

vector<ColumnBinding> LogicalVectorIndexJoin::GetLeftBindings() {
	vector<ColumnBinding> result;
	if (inner_projection_ids.empty()) {
		for (idx_t col_idx = 0; col_idx < inner_column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for (auto proj_id : inner_projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}
	result.emplace_back(table_index, inner_column_ids.size());
	return result;
}

vector<ColumnBinding> LogicalVectorIndexJoin::GetRightBindings() {
	vector<ColumnBinding> result;
	for (auto &binding : children[0]->GetColumnBindings()) {
		result.push_back(binding);
	}
	return result;
}

vector<ColumnBinding> LogicalVectorIndexJoin::GetColumnBindings() {
	vector<ColumnBinding> result;
	auto left_bindings = GetLeftBindings();
	auto right_bindings = GetRightBindings();
	result.insert(result.end(), left_bindings.begin(), left_bindings.end());
	result.insert(result.end(), right_bindings.begin(), right_bindings.end());
	return result;
}

PhysicalOperator &LogicalVectorIndexJoin::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	auto &result = planner.Make<PhysicalVectorIndexJoin>(types, estimated_cardinality, table, index, limit);
	auto &cast_result = result.Cast<PhysicalVectorIndexJoin>();
	cast_result.limit = limit;
	cast_result.inner_column_ids = inner_column_ids;
	cast_result.inner_projection_ids = inner_projection_ids;
	cast_result.outer_vector_column = outer_vector_column;
	cast_result.inner_vector_column = inner_vector_column;

	auto &plan = planner.CreatePlan(*children[0]);
	result.children.push_back(plan);
	return result;
}

idx_t LogicalVectorIndexJoin::EstimateCardinality(ClientContext &context) {
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}
	const auto child_cardinality = children[0]->EstimateCardinality(context);
	estimated_cardinality = child_cardinality * limit;
	has_estimated_cardinality = true;
	return estimated_cardinality;
}

//------------------------------------------------------------------------------
// Optimizer
//------------------------------------------------------------------------------

class VectorIndexJoinOptimizer : public OptimizerExtension {
public:
	VectorIndexJoinOptimizer() {
		optimize_function = Optimize;
	}
	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root,
	                        unique_ptr<LogicalOperator> &plan);
	static void OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root,
	                              unique_ptr<LogicalOperator> &plan);
	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

class CardinalityResetter final : public LogicalOperatorVisitor {
public:
	ClientContext &context;
	explicit CardinalityResetter(ClientContext &context_p) : context(context_p) {
	}
	void VisitOperator(LogicalOperator &op) override {
		op.has_estimated_cardinality = false;
		VisitOperatorChildren(op);
		op.EstimateCardinality(context);
	}
};

bool VectorIndexJoinOptimizer::TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root,
                                           unique_ptr<LogicalOperator> &plan) {
#define MATCH_OPERATOR(OP, TYPE, CHILD_COUNT)                                                                          \
	if (OP->type != LogicalOperatorType::TYPE || (OP->children.size() != CHILD_COUNT)) {                               \
		return false;                                                                                                  \
	}

	MATCH_OPERATOR(plan, LOGICAL_DELIM_JOIN, 2);
	auto &delim_join = plan->Cast<LogicalJoin>();

	MATCH_OPERATOR(delim_join.children[1], LOGICAL_GET, 0);
	auto outer_get_ptr = &delim_join.children[1];
	auto &outer_get = (*outer_get_ptr)->Cast<LogicalGet>();

	const unique_ptr<LogicalOperator> *filter_ptr = nullptr;
	const unique_ptr<LogicalOperator> *outer_proj_ptr = nullptr;

	auto &delim_child = delim_join.children[0];
	if (delim_child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &filter_proj = delim_child->Cast<LogicalProjection>();
		if (filter_proj.children.back()->type != LogicalOperatorType::LOGICAL_FILTER) {
			return false;
		}
		outer_proj_ptr = &delim_child;
		filter_ptr = &filter_proj.children.back();
	} else if (delim_child->type == LogicalOperatorType::LOGICAL_FILTER) {
		filter_ptr = &delim_child;
	} else {
		return false;
	}

	auto &filter = (*filter_ptr)->Cast<LogicalFilter>();
	MATCH_OPERATOR(filter.children.back(), LOGICAL_WINDOW, 1);
	auto &window = filter.children.back()->Cast<LogicalWindow>();

	MATCH_OPERATOR(window.children[0], LOGICAL_PROJECTION, 1);
	auto &inner_proj = window.children[0]->Cast<LogicalProjection>();

	MATCH_OPERATOR(inner_proj.children[0], LOGICAL_CROSS_PRODUCT, 2);
	auto &cross_product = inner_proj.children[0]->Cast<LogicalCrossProduct>();
#undef MATCH_OPERATOR

	unique_ptr<LogicalOperator> *delim_get_ptr;
	unique_ptr<LogicalOperator> *inner_get_ptr;

	auto &cp_lhs = cross_product.children[0];
	auto &cp_rhs = cross_product.children[1];
	if (cp_lhs->type == LogicalOperatorType::LOGICAL_DELIM_GET && cp_rhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_lhs;
		inner_get_ptr = &cp_rhs;
	} else if (cp_rhs->type == LogicalOperatorType::LOGICAL_DELIM_GET &&
	           cp_lhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_rhs;
		inner_get_ptr = &cp_lhs;
	} else {
		return false;
	}

	auto &delim_get = (*delim_get_ptr)->Cast<LogicalDelimGet>();
	auto &inner_get = (*inner_get_ptr)->Cast<LogicalGet>();
	if (inner_get.function.name != "seq_scan") {
		return false;
	}

	if (filter.expressions.size() != 1) {
		return false;
	}
	if (filter.expressions.back()->type != ExpressionType::COMPARE_LESSTHANOREQUALTO) {
		return false;
	}
	const auto &compare_expr = filter.expressions.back()->Cast<BoundComparisonExpression>();
	if (compare_expr.right->type != ExpressionType::VALUE_CONSTANT) {
		return false;
	}
	const auto &constant_expr = compare_expr.right->Cast<BoundConstantExpression>();
	if (constant_expr.return_type != LogicalType::BIGINT) {
		return false;
	}
	auto k_value = constant_expr.value.GetValue<int64_t>();
	if (k_value < 0 || k_value >= STANDARD_VECTOR_SIZE) {
		return false;
	}
	if (compare_expr.left->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &filter_ref_expr = compare_expr.left->Cast<BoundColumnRefExpression>();
	if (filter_ref_expr.binding.table_index != window.window_index) {
		return false;
	}
	if (filter_ref_expr.binding.column_index != 0) {
		return false;
	}

	if (window.expressions.size() != 1) {
		return false;
	}
	if (window.expressions.back()->type != ExpressionType::WINDOW_ROW_NUMBER) {
		return false;
	}
	auto &window_expr = window.expressions.back()->Cast<BoundWindowExpression>();
	if (window_expr.orders.size() != 1) {
		return false;
	}
	if (window_expr.orders.back().type != OrderType::ASCENDING) {
		return false;
	}
	if (window_expr.orders.back().expression->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	const auto &distance_ref_expr = window_expr.orders.back().expression->Cast<BoundColumnRefExpression>();
	if (distance_ref_expr.binding.table_index != inner_proj.table_index) {
		return false;
	}
	if (distance_ref_expr.binding.column_index >= inner_proj.expressions.size()) {
		return false;
	}
	const auto &distance_expr_ptr = inner_proj.expressions[distance_ref_expr.binding.column_index];

	auto &table = *inner_get.GetTable();
	if (!table.IsDuckTable()) {
		return false;
	}
	auto &duck_table = table.Cast<DuckTableEntry>();
	auto &table_info = *table.GetStorage().GetDataTableInfo();

	VectorIndex *index_ptr = nullptr;
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
		if (!vi->TryMatchDistanceFunction(distance_expr_ptr, bindings)) {
			continue;
		}
		unique_ptr<Expression> bound_index_expr;
		if (!vi->TryBindIndexExpression(inner_get, bound_index_expr)) {
			continue;
		}

		ExpressionIterator::EnumerateExpression(bound_index_expr, [&](Expression &child) {
			if (child.type == ExpressionType::BOUND_COLUMN_REF) {
				auto &bound_colref_expr = child.Cast<BoundColumnRefExpression>();
				if (bound_colref_expr.binding.table_index == outer_get.table_index) {
					bound_colref_expr.binding.table_index = delim_get.table_index;
				}
			}
		});

		auto &lhs_dist_expr = bindings[1];
		auto &rhs_dist_expr = bindings[2];

		if (lhs_dist_expr.get().Equals(*bound_index_expr)) {
			if (!rhs_dist_expr.get().Equals(*bound_index_expr)) {
				std::swap(lhs_dist_expr, rhs_dist_expr);
			} else {
				continue;
			}
		}

		index_ptr = vi;
		break;
	}
	if (!index_ptr) {
		return false;
	}

	if (bindings[1].get().type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	if (bindings[2].get().type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	const auto &outer_ref_expr = bindings[1].get().Cast<BoundColumnRefExpression>();
	const auto &inner_ref_expr = bindings[2].get().Cast<BoundColumnRefExpression>();

	if (inner_ref_expr.binding.table_index != inner_get.table_index) {
		return false;
	}

	auto index_join =
	    make_uniq<LogicalVectorIndexJoin>(binder.GenerateTableIndex(), duck_table, *index_ptr, k_value);
	for (auto &column_id : inner_get.GetColumnIds()) {
		index_join->inner_column_ids.emplace_back(column_id.GetPrimaryIndex());
	}
	index_join->inner_projection_ids = inner_get.projection_ids;
	index_join->inner_returned_types = inner_get.returned_types;
	index_join->outer_vector_column = outer_ref_expr.binding.column_index;
	index_join->inner_vector_column = inner_ref_expr.binding.column_index;

	ColumnBindingReplacer replacer;
	vector<unique_ptr<Expression>> projection_expressions;
	auto projection_table_index = binder.GenerateTableIndex();
	auto delim_bindings = delim_join.GetColumnBindings();
	auto delim_types = delim_join.types;

	idx_t new_binding_idx = 0;
	for (idx_t i = 0; i < delim_bindings.size(); i++) {
		auto &old_binding = delim_bindings[i];
		if (old_binding.table_index == window.window_index) {
			continue;
		}
		auto &old_type = delim_types[i];
		projection_expressions.push_back(make_uniq<BoundColumnRefExpression>(old_type, old_binding));
		replacer.replacement_bindings.emplace_back(old_binding,
		                                           ColumnBinding(projection_table_index, new_binding_idx++));
	}

	ColumnBinding window_binding(window.window_index, 0);
	projection_expressions.push_back(make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, window_binding));
	replacer.replacement_bindings.emplace_back(window_binding,
	                                           ColumnBinding(projection_table_index, new_binding_idx++));

	auto new_projection = make_uniq<LogicalProjection>(projection_table_index, std::move(projection_expressions));
	replacer.VisitOperator(*root);
	replacer.replacement_bindings.clear();

	for (auto &expr : new_projection->expressions) {
		auto &ref = expr->Cast<BoundColumnRefExpression>();

		if (ref.binding.table_index == inner_get.table_index) {
			ref.binding.table_index = index_join->table_index;
		}
		if (outer_proj_ptr) {
			auto &outer_proj = outer_proj_ptr->get()->Cast<LogicalProjection>();
			if (ref.binding.table_index == outer_proj.table_index) {
				const auto &outer_expr = outer_proj.expressions[ref.binding.column_index];
				const auto &outer_ref = outer_expr->Cast<BoundColumnRefExpression>();
				ref.binding = outer_ref.binding;
			}
		}
		if (ref.binding.table_index == inner_proj.table_index) {
			const auto &inner_expr = inner_proj.expressions[ref.binding.column_index];
			expr = inner_expr->Copy();
		} else if (ref.binding.table_index == window.window_index) {
			ColumnBinding index_row_number_binding(index_join->table_index, index_join->inner_column_ids.size());
			expr = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, index_row_number_binding);
		}
	}

	for (const auto &old_binding : delim_get.GetColumnBindings()) {
		replacer.replacement_bindings.emplace_back(old_binding,
		                                           ColumnBinding(outer_get.table_index, old_binding.column_index));
	}
	for (const auto &old_binding : inner_get.GetColumnBindings()) {
		replacer.replacement_bindings.emplace_back(old_binding,
		                                           ColumnBinding(index_join->table_index, old_binding.column_index));
	}
	replacer.VisitOperator(*new_projection);

	index_join->children.emplace_back(std::move(*outer_get_ptr));
	new_projection->children.emplace_back(std::move(index_join));
	new_projection->EstimateCardinality(context);
	plan = std::move(new_projection);

	CardinalityResetter cardinality_resetter(context);
	cardinality_resetter.VisitOperator(*root);
	return true;
}

void VectorIndexJoinOptimizer::OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root,
                                                 unique_ptr<LogicalOperator> &plan) {
	if (!TryOptimize(input.optimizer.binder, input.context, root, plan)) {
		for (auto &child : plan->children) {
			OptimizeRecursive(input, root, child);
		}
	}
}

void VectorIndexJoinOptimizer::Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	OptimizeRecursive(input, plan, plan);
}

void RegisterJoinOptimizer(DatabaseInstance &db) {
	ExtensionCallbackManager::Get(db).Register(VectorIndexJoinOptimizer());
}

} // namespace vindex
} // namespace duckdb
