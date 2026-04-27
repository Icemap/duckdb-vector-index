#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/optimizer/expression_rewriter.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/optimizer/matcher/function_matcher.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/rule.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// Expression-rewrite rule: (1.0 - array_cosine_similarity(a, b))
//                       => array_cosine_distance(a, b)
//
// Algorithm-agnostic: works on arbitrary vector distance expressions so that
// downstream optimizers can match `array_cosine_distance` directly.
// ---------------------------------------------------------------------------

class CosineDistanceRule final : public Rule {
public:
	explicit CosineDistanceRule(ExpressionRewriter &rewriter);
	unique_ptr<Expression> Apply(LogicalOperator &op, vector<reference<Expression>> &bindings, bool &changes_made,
	                             bool is_root) override;
};

CosineDistanceRule::CosineDistanceRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto func = make_uniq<FunctionExpressionMatcher>();
	func->matchers.push_back(make_uniq<ExpressionMatcher>());
	func->matchers.push_back(make_uniq<ExpressionMatcher>());
	func->policy = SetMatcher::Policy::UNORDERED;
	func->function = make_uniq<SpecificFunctionMatcher>("array_cosine_similarity");

	auto op = make_uniq<FunctionExpressionMatcher>();
	op->matchers.push_back(make_uniq<ConstantExpressionMatcher>());
	op->matchers[0]->type = make_uniq<SpecificTypeMatcher>(LogicalType::FLOAT);
	op->matchers.push_back(std::move(func));
	op->policy = SetMatcher::Policy::ORDERED;
	op->function = make_uniq<SpecificFunctionMatcher>("-");
	op->type = make_uniq<SpecificTypeMatcher>(LogicalType::FLOAT);

	root = std::move(op);
}

unique_ptr<Expression> CosineDistanceRule::Apply(LogicalOperator &op, vector<reference<Expression>> &bindings,
                                                 bool &changes_made, bool is_root) {
	const auto &const_expr = bindings[1].get().Cast<BoundConstantExpression>();
	auto &similarity_expr = bindings[2].get().Cast<BoundFunctionExpression>();

	if (!const_expr.value.IsNull() && const_expr.value.GetValue<float>() == 1.0) {
		vector<unique_ptr<Expression>> args;
		vector<LogicalType> arg_types;
		arg_types.push_back(similarity_expr.children[0]->return_type);
		arg_types.push_back(similarity_expr.children[1]->return_type);
		args.push_back(std::move(similarity_expr.children[0]));
		args.push_back(std::move(similarity_expr.children[1]));

		auto &context = GetContext();
		auto func_entry = Catalog::GetEntry<ScalarFunctionCatalogEntry>(context, "", "", "array_cosine_distance",
		                                                                OnEntryNotFound::RETURN_NULL);
		if (!func_entry) {
			return nullptr;
		}

		changes_made = true;
		auto func = func_entry->functions.GetFunctionByArguments(context, arg_types);
		return make_uniq<BoundFunctionExpression>(similarity_expr.return_type, func, std::move(args), nullptr);
	}
	return nullptr;
}

class VectorExprOptimizer : public OptimizerExtension {
public:
	VectorExprOptimizer() {
		optimize_function = Optimize;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		ExpressionRewriter rewriter(input.context);
		rewriter.rules.push_back(make_uniq<CosineDistanceRule>(rewriter));
		rewriter.VisitOperator(*plan);
	}
};

void RegisterExprOptimizer(DatabaseInstance &db) {
	ExtensionCallbackManager::Get(db).Register(VectorExprOptimizer());
}

} // namespace vindex
} // namespace duckdb
