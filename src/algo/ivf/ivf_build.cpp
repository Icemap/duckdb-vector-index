#include "algo/ivf/ivf_build.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "algo/ivf/ivf_index.hpp"

namespace duckdb {
namespace vindex {
namespace ivf {

PhysicalCreateIvfIndex::PhysicalCreateIvfIndex(PhysicalPlan &physical_plan, const vector<LogicalType> &types_p,
                                               TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                               unique_ptr<CreateIndexInfo> info,
                                               vector<unique_ptr<Expression>> unbound_expressions,
                                               idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types_p, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

//-------------------------------------------------------------
// Global / Local State
//-------------------------------------------------------------
class CreateIvfIndexGlobalState final : public GlobalSinkState {
public:
	explicit CreateIvfIndexGlobalState(const PhysicalOperator &op_p) : op(op_p) {
	}

	const PhysicalOperator &op;
	unique_ptr<IvfIndex> global_index;

	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;

	ColumnDataParallelScanState scan_state;

	atomic<bool> is_building {false};
	atomic<idx_t> loaded_count {0};
	atomic<idx_t> built_count {0};
};

unique_ptr<GlobalSinkState> PhysicalCreateIvfIndex::GetGlobalSinkState(ClientContext &context) const {
	auto gstate = make_uniq<CreateIvfIndexGlobalState>(*this);

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	gstate->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), data_types);
	gstate->context = context.shared_from_this();

	auto &storage = table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = info->constraint_type;
	auto &db = storage.db;
	gstate->global_index =
	    make_uniq<IvfIndex>(info->index_name, constraint_type, storage_ids, table_manager, unbound_expressions, db,
	                        info->options, IndexStorageInfo(), estimated_cardinality);

	return std::move(gstate);
}

class CreateIvfIndexLocalState final : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

unique_ptr<LocalSinkState> PhysicalCreateIvfIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateIvfIndexLocalState>();
	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context.client), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

SinkResultType PhysicalCreateIvfIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<CreateIvfIndexLocalState>();
	auto &gstate = input.global_state.Cast<CreateIvfIndexGlobalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	gstate.loaded_count += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreateIvfIndex::Combine(ExecutionContext &context,
                                                      OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<CreateIvfIndexGlobalState>();
	auto &lstate = input.local_state.Cast<CreateIvfIndexLocalState>();

	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}
	return SinkCombineResultType::FINISHED;
}

//-------------------------------------------------------------
// Finalize
//-------------------------------------------------------------

class IvfIndexConstructTask final : public ExecutorTask {
public:
	IvfIndexConstructTask(shared_ptr<Event> event_p, ClientContext &context, CreateIvfIndexGlobalState &gstate_p,
	                      size_t thread_id_p, const PhysicalCreateIvfIndex &op_p)
	    : ExecutorTask(context, std::move(event_p), op_p), gstate(gstate_p), thread_id(thread_id_p), local_scan_state() {
		gstate.collection->InitializeScanChunk(scan_chunk);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		auto &scan_state = gstate.scan_state;
		auto &collection = gstate.collection;

		DataChunk build_chunk;
		build_chunk.Initialize(executor.context, {scan_chunk.data[0].GetType()});
		Vector row_ids(LogicalType::ROW_TYPE);

		// IvfIndex::Construct serializes through rwlock. Keeping multiple
		// construct tasks is primarily for future benefit when assignment can
		// be parallelized; currently one thread does all useful work.
		while (collection->Scan(scan_state, local_scan_state, scan_chunk)) {
			const auto count = scan_chunk.size();
			build_chunk.Reset();
			build_chunk.data[0].Reference(scan_chunk.data[0]);
			build_chunk.SetCardinality(count);
			row_ids.Reference(scan_chunk.data[1]);

			gstate.global_index->Construct(build_chunk, row_ids, thread_id);
			gstate.built_count += count;

			if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}
		}

		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	CreateIvfIndexGlobalState &gstate;
	size_t thread_id;

	DataChunk scan_chunk;
	ColumnDataLocalScanState local_scan_state;
};

class IvfIndexConstructionEvent final : public BasePipelineEvent {
public:
	IvfIndexConstructionEvent(const PhysicalCreateIvfIndex &op_p, CreateIvfIndexGlobalState &gstate_p,
	                          Pipeline &pipeline_p, CreateIndexInfo &info_p, const vector<column_t> &storage_ids_p,
	                          DuckTableEntry &table_p)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), info(info_p), storage_ids(storage_ids_p),
	      table(table_p) {
	}

	const PhysicalCreateIvfIndex &op;
	CreateIvfIndexGlobalState &gstate;
	CreateIndexInfo &info;
	const vector<column_t> &storage_ids;
	DuckTableEntry &table;

	void Schedule() override {
		auto &context = pipeline->GetClientContext();
		auto &ts = TaskScheduler::GetScheduler(context);
		const auto num_threads = NumericCast<size_t>(ts.NumberOfThreads());

		vector<shared_ptr<Task>> construct_tasks;
		for (size_t tnum = 0; tnum < num_threads; tnum++) {
			construct_tasks.push_back(make_uniq<IvfIndexConstructTask>(shared_from_this(), context, gstate, tnum, op));
		}
		SetTasks(std::move(construct_tasks));
	}

	void FinishEvent() override {
		gstate.global_index->SetDirty();
		gstate.global_index->SyncSize();

		auto &storage = table.GetStorage();
		if (!storage.db.GetStorageManager().InMemory()) {
			gstate.global_index->PersistToDisk();
		}

		if (!storage.IsRoot()) {
			throw TransactionException("Cannot create index on non-root transaction");
		}

		auto &schema = table.schema;
		info.column_ids = storage_ids;

		if (schema.GetEntry(schema.GetCatalogTransaction(*gstate.context), CatalogType::INDEX_ENTRY, info.index_name)) {
			if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
				throw CatalogException("Index with name \"%s\" already exists", info.index_name);
			}
		}

		const auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(*gstate.context), info, table).get();
		D_ASSERT(index_entry);
		auto &duck_index = index_entry->Cast<DuckIndexEntry>();
		duck_index.initial_index_size = gstate.global_index->Cast<BoundIndex>().GetInMemorySize();

		storage.AddIndex(std::move(gstate.global_index));
	}
};

SinkFinalizeType PhysicalCreateIvfIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<CreateIvfIndexGlobalState>();
	auto &collection = gstate.collection;

	gstate.is_building = true;

	// Two training passes before the parallel Construct phase.
	// Order matters: quantizer trains first because IvfCore::Insert encodes
	// via the quantizer (both RaBitQ and Flat must be ready). Centroids
	// second — they only need float samples, but sharing the pass simplifies
	// the bookkeeping.
	gstate.global_index->TrainQuantizer(*collection);
	gstate.global_index->TrainCentroids(*collection);

	collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	auto new_event = make_shared_ptr<IvfIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);
	event.InsertEvent(std::move(new_event));
	return SinkFinalizeType::READY;
}

ProgressData PhysicalCreateIvfIndex::GetSinkProgress(ClientContext &context, GlobalSinkState &gstate,
                                                     ProgressData source_progress) const {
	ProgressData res;
	const auto &state = gstate.Cast<CreateIvfIndexGlobalState>();
	if (!state.is_building) {
		res.done = state.loaded_count + 0.0;
		res.total = estimated_cardinality + estimated_cardinality;
	} else {
		res.done = state.loaded_count + state.built_count;
		res.total = state.loaded_count + state.loaded_count;
	}
	return res;
}

} // namespace ivf
} // namespace vindex
} // namespace duckdb
