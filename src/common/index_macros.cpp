#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include "vindex/vector_index_registry.hpp"

namespace duckdb {
namespace vindex {

// Ported from vss's hnsw_index_macros.cpp. The macros are algorithm-agnostic
// SQL (they just branch on a `metric` argument), so they live under
// src/common/ and are registered once by the extension loader.
//
// Registered under the `vindex_*` names; the legacy `vss_*` names are
// kept as aliases for one minor release to smooth vss migration.

static constexpr auto VSS_JOIN_MACRO = R"(
SELECT
	score,
	left_tbl,
	right_tbl,
FROM
	(SELECT * FROM query_table(left_table::VARCHAR)) as left_tbl,
	(
		SELECT
			struct_pack(*columns([x for x in (matches.*) if x != 'score'])) as right_tbl,
			matches.score as score
		FROM (
			SELECT (
				unnest(
					CASE WHEN metric = 'l2sq' OR metric = 'l2'
						THEN min_by(tbl, tbl.score, k)
						ELSE max_by(tbl, tbl.score, k)
					END,
					max_depth := 2
				)
			) as result
			FROM (
				SELECT
				*,
				CASE
					WHEN metric = 'l2sq' OR metric = 'l2'
					THEN array_distance(left_col, right_col)
					WHEN metric = 'cosine' OR metric = 'cos'
					THEN array_cosine_similarity(left_col, right_col)
					WHEN metric = 'ip'
					THEN array_inner_product(left_col, right_col)
					ELSE error('Unknown metric')
				END as score,
				FROM query_table(right_table::VARCHAR)
			) as tbl
		) as matches
	)
)";

static constexpr auto VSS_MATCH_MACRO = R"(
SELECT
	right_tbl as matches,
FROM
	(
		SELECT (
			CASE WHEN metric = 'l2sq' OR metric = 'l2'
				THEN min_by({'score': score, 'row': t}, score, k)
				ELSE max_by({'score': score, 'row': t}, score, k)
			END
		) as right_tbl
		FROM (
			SELECT
			CASE
				WHEN metric = 'l2sq' OR metric = 'l2'
				THEN array_distance(left_col, right_col)
				WHEN metric = 'cosine' OR metric = 'cos'
				THEN array_cosine_similarity(left_col, right_col)
				WHEN metric = 'ip'
				THEN array_inner_product(left_col, right_col)
				ELSE error('Unknown metric')
			END as score,
			tbl as t,
			FROM (SELECT * FROM query_table(right_table::VARCHAR)) as tbl
		)
	)
)";

static void RegisterTableMacro(ExtensionLoader &loader, const string &name, const string &query,
                               const vector<string> &params, const child_list_t<Value> &named_params) {
	Parser parser;
	parser.ParseQuery(query);
	const auto &stmt = parser.statements.back();
	auto &node = stmt->Cast<SelectStatement>().node;

	auto func = make_uniq<TableMacroFunction>(std::move(node));
	for (auto &param : params) {
		func->parameters.push_back(make_uniq<ColumnRefExpression>(param));
	}
	for (auto &param : named_params) {
		func->parameters.push_back(make_uniq<ColumnRefExpression>(param.first));
		func->default_parameters[param.first] = make_uniq<ConstantExpression>(param.second);
	}

	CreateMacroInfo info(CatalogType::TABLE_MACRO_ENTRY);
	info.schema = DEFAULT_SCHEMA;
	info.name = name;
	info.temporary = true;
	info.internal = true;
	info.macros.push_back(std::move(func));

	loader.RegisterFunction(info);
}

void RegisterMacros(ExtensionLoader &loader) {
	// New-world names.
	RegisterTableMacro(loader, "vindex_join", VSS_JOIN_MACRO,
	                   {"left_table", "right_table", "left_col", "right_col", "k"}, {{"metric", Value("l2sq")}});
	RegisterTableMacro(loader, "vindex_match", VSS_MATCH_MACRO, {"right_table", "left_col", "right_col", "k"},
	                   {{"metric", Value("l2sq")}});

	// Legacy aliases — same macro body, vss_* names. Kept for one minor
	// release so we don't break existing users of vss.
	RegisterTableMacro(loader, "vss_join", VSS_JOIN_MACRO,
	                   {"left_table", "right_table", "left_col", "right_col", "k"}, {{"metric", Value("l2sq")}});
	RegisterTableMacro(loader, "vss_match", VSS_MATCH_MACRO, {"right_table", "left_col", "right_col", "k"},
	                   {{"metric", Value("l2sq")}});
}

} // namespace vindex
} // namespace duckdb
