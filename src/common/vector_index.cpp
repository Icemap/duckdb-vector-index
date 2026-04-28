#include "vindex/vector_index.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/optimizer/matcher/expression_type_matcher.hpp"
#include "duckdb/optimizer/matcher/function_matcher.hpp"
#include "duckdb/optimizer/matcher/set_matcher.hpp"
#include "duckdb/optimizer/matcher/type_matcher.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {
namespace vindex {

VectorIndex::VectorIndex(const string &name, const string &type_name, IndexConstraintType constraint_type,
                         const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                         const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db)
    : BoundIndex(name, type_name, constraint_type, column_ids, table_io_manager, unbound_expressions, db) {
}

unique_ptr<ExpressionMatcher> VectorIndex::BuildDistanceMatcher() const {
	auto names = DistanceFunctionNames(GetMetricKind());
	unordered_set<string> fns(names.begin(), names.end());

	auto matcher = make_uniq<FunctionExpressionMatcher>();
	matcher->function = make_uniq<ManyFunctionMatcher>(std::move(fns));
	matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	matcher->policy = SetMatcher::Policy::UNORDERED;

	auto lhs = make_uniq<ExpressionMatcher>();
	lhs->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, GetVectorSize()));
	matcher->matchers.push_back(std::move(lhs));

	auto rhs = make_uniq<ExpressionMatcher>();
	rhs->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, GetVectorSize()));
	matcher->matchers.push_back(std::move(rhs));
	return std::move(matcher);
}

bool VectorIndex::TryMatchDistanceFunction(const unique_ptr<Expression> &expr,
                                           vector<reference<Expression>> &bindings) const {
	if (!function_matcher_) {
		function_matcher_ = BuildDistanceMatcher();
	}
	return function_matcher_->Match(*expr, bindings);
}

static void RewriteInternal(Expression &expr, idx_t table_idx, const vector<column_t> &index_columns,
                            const vector<ColumnIndex> &table_columns, bool &success, bool &found) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		found = true;
		auto &ref = expr.Cast<BoundColumnRefExpression>();
		ref.binding.table_index = table_idx;
		const auto target = index_columns[ref.binding.column_index];
		for (idx_t i = 0; i < table_columns.size(); i++) {
			if (table_columns[i].GetPrimaryIndex() == target) {
				ref.binding.column_index = i;
				return;
			}
		}
		success = false;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { RewriteInternal(child, table_idx, index_columns, table_columns, success, found); });
}

bool VectorIndex::TryBindIndexExpression(LogicalGet &get, unique_ptr<Expression> &result) const {
	auto expr = unbound_expressions.back()->Copy();
	bool success = true;
	bool found = false;
	RewriteInternal(*expr, get.table_index, GetColumnIds(), get.GetColumnIds(), success, found);
	if (success && found) {
		result = std::move(expr);
		return true;
	}
	return false;
}

} // namespace vindex
} // namespace duckdb
