#pragma once

#include "duckdb.hpp"

namespace duckdb {

class VindexExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
};

} // namespace duckdb
