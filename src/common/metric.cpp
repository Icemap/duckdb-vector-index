#include "vindex/metric.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace vindex {

MetricKind ParseMetric(const string &name) {
	if (StringUtil::CIEquals(name, "l2sq")) {
		return MetricKind::L2SQ;
	}
	if (StringUtil::CIEquals(name, "cosine")) {
		return MetricKind::COSINE;
	}
	if (StringUtil::CIEquals(name, "ip")) {
		return MetricKind::IP;
	}
	throw BinderException("vindex: unknown metric '%s' (expected l2sq|cosine|ip)", name);
}

const char *MetricName(MetricKind kind) {
	switch (kind) {
	case MetricKind::L2SQ:
		return "l2sq";
	case MetricKind::COSINE:
		return "cosine";
	case MetricKind::IP:
		return "ip";
	}
	throw InternalException("vindex: unreachable MetricKind");
}

vector<string> DistanceFunctionNames(MetricKind kind) {
	switch (kind) {
	case MetricKind::L2SQ:
		return {"array_distance", "<->"};
	case MetricKind::COSINE:
		return {"array_cosine_distance", "<=>"};
	case MetricKind::IP:
		return {"array_negative_inner_product", "<#>"};
	}
	throw InternalException("vindex: unreachable MetricKind");
}

} // namespace vindex
} // namespace duckdb
