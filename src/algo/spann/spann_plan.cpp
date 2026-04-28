#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "algo/spann/spann_build.hpp"
#include "algo/spann/spann_index.hpp"

namespace duckdb {
namespace vindex {
namespace spann {

PhysicalOperator &SpannIndex::CreatePlan(PlanIndexInput &input) {
	auto &create_index = input.op;
	auto &context = input.context;
	auto &planner = input.planner;

	Value vindex_val, hnsw_val;
	context.TryGetCurrentSetting("vindex_enable_experimental_persistence", vindex_val);
	context.TryGetCurrentSetting("hnsw_enable_experimental_persistence", hnsw_val);
	const bool enable_persistence = (!vindex_val.IsNull() && vindex_val.GetValue<bool>()) ||
	                                (!hnsw_val.IsNull() && hnsw_val.GetValue<bool>());

	auto is_disk_db = !create_index.table.GetStorage().db.GetStorageManager().InMemory();
	auto is_persistence_disabled = !enable_persistence;

	if (is_disk_db && is_persistence_disabled) {
		throw BinderException("SPANN indexes can only be created in in-memory databases, or when the configuration "
		                      "option 'vindex_enable_experimental_persistence' is set to true.");
	}

	for (auto &option : create_index.info->options) {
		auto &k = option.first;
		auto &v = option.second;
		if (StringUtil::CIEquals(k, "metric")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("SPANN index 'metric' must be a string");
			}
			const auto metric = StringUtil::Lower(v.GetValue<string>());
			if (metric != "l2sq" && metric != "cosine" && metric != "ip") {
				throw BinderException("SPANN index 'metric' must be one of: 'l2sq', 'cosine', 'ip'");
			}
		} else if (StringUtil::CIEquals(k, "quantizer")) {
			// Validated by the quantizer factory.
		} else if (StringUtil::CIEquals(k, "bits")) {
			if (v.type() != LogicalType::INTEGER) {
				throw BinderException("SPANN index 'bits' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "m")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("SPANN index 'm' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "eta")) {
			// ScaNN anisotropic loss ratio; validated by factory.
		} else if (StringUtil::CIEquals(k, "rerank")) {
			if (v.type() != LogicalType::INTEGER) {
				throw BinderException("SPANN index 'rerank' must be an integer");
			}
			if (v.GetValue<int32_t>() < 1) {
				throw BinderException("SPANN index 'rerank' must be at least 1");
			}
		} else if (StringUtil::CIEquals(k, "nlist")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("SPANN index 'nlist' must be an integer");
			}
			if (v.GetValue<int64_t>() < 2) {
				throw BinderException("SPANN index 'nlist' must be at least 2");
			}
		} else if (StringUtil::CIEquals(k, "nprobe")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("SPANN index 'nprobe' must be an integer");
			}
			if (v.GetValue<int64_t>() < 1) {
				throw BinderException("SPANN index 'nprobe' must be at least 1");
			}
		} else if (StringUtil::CIEquals(k, "replica_count")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("SPANN index 'replica_count' must be an integer");
			}
			if (v.GetValue<int64_t>() < 1) {
				throw BinderException("SPANN index 'replica_count' must be at least 1");
			}
		} else if (StringUtil::CIEquals(k, "closure_factor")) {
			// Full numeric validation happens in SpannIndex ctor (uses
			// DefaultTryCastAs<DOUBLE> to accept DECIMAL literals like 1.2).
		} else {
			throw BinderException("Unknown option for SPANN index: '%s'", k);
		}
	}

	if (create_index.expressions.size() != 1) {
		throw BinderException("SPANN indexes can only be created over a single column of keys.");
	}
	auto &arr_type = create_index.expressions[0]->return_type;
	if (arr_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("SPANN index keys must be of type FLOAT[N]");
	}
	auto &child_type = ArrayType::GetChildType(arr_type);
	if (child_type.id() != LogicalTypeId::FLOAT) {
		throw BinderException("SPANN index key type must be 'FLOAT[N]'");
	}

	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (auto &expression : create_index.expressions) {
		new_column_types.push_back(expression->return_type);
		select_list.push_back(std::move(expression));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(
	    make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, create_index.info->scan_types.size() - 1));

	auto &projection =
	    planner.Make<PhysicalProjection>(new_column_types, std::move(select_list), create_index.estimated_cardinality);
	projection.children.push_back(input.table_scan);

	vector<LogicalType> filter_types;
	vector<unique_ptr<Expression>> filter_select_list;

	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		filter_types.push_back(new_column_types[i]);
		auto is_not_null_expr =
		    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
		auto bound_ref = make_uniq<BoundReferenceExpression>(new_column_types[i], i);
		is_not_null_expr->children.push_back(std::move(bound_ref));
		filter_select_list.push_back(std::move(is_not_null_expr));
	}

	auto &null_filter = planner.Make<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list),
	                                                 create_index.estimated_cardinality);
	null_filter.types.emplace_back(LogicalType::ROW_TYPE);
	null_filter.children.push_back(projection);

	auto &physical_create_index = planner.Make<PhysicalCreateSpannIndex>(
	    create_index.types, create_index.table, create_index.info->column_ids, std::move(create_index.info),
	    std::move(create_index.unbound_expressions), create_index.estimated_cardinality);
	physical_create_index.children.push_back(null_filter);
	return physical_create_index;
}

} // namespace spann
} // namespace vindex
} // namespace duckdb
