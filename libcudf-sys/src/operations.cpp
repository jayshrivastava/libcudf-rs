#include "operations.h"
#include "libcudf-sys/src/lib.rs.h"

#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/version_config.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <nanoarrow/nanoarrow.h>

#include <memory>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>

namespace libcudf_bridge {
    // Factory functions
    std::unique_ptr<Table> create_empty_table() {
        auto table = std::make_unique<Table>();
        std::vector<std::unique_ptr<cudf::column> > columns;
        table->inner = std::make_unique<cudf::table>(std::move(columns));
        return table;
    }

    std::unique_ptr<Table> create_table_from_columns_move(rust::Slice<Column *const> columns) {
        std::vector<std::unique_ptr<cudf::column> > cudf_columns;
        cudf_columns.reserve(columns.size());

        // Take ownership of columns by moving from each pointer
        for (auto *col: columns) {
            cudf_columns.push_back(std::move(col->inner));
        }

        auto table = std::make_unique<Table>();
        table->inner = std::make_unique<cudf::table>(std::move(cudf_columns));
        return table;
    }

    std::unique_ptr<Table> concat_table_views(rust::Slice<const std::unique_ptr<TableView>> views) {
        std::vector<cudf::table_view> table_views;
        table_views.reserve(views.size());

        // Take ownership of tables by moving out of each unique pointer
        for (auto &col: views) {
            table_views.push_back(std::move(*col->inner));
        }

        auto table = std::make_unique<Table>();
        table->inner = cudf::concatenate(table_views);
        return table;
    }

    std::unique_ptr<Table> concat_table_views_on(
        rust::Slice<const std::unique_ptr<TableView>> views,
        const CudaStream &stream) {
        std::vector<cudf::table_view> table_views;
        table_views.reserve(views.size());
        for (auto &col: views) {
            table_views.push_back(std::move(*col->inner));
        }
        auto table = std::make_unique<Table>();
        table->inner = cudf::concatenate(table_views, stream.view());
        return table;
    }

    std::unique_ptr<Column> concat_column_views(rust::Slice<const std::unique_ptr<ColumnView>> views) {
        std::vector<cudf::column_view> table_views;
        table_views.reserve(views.size());

        // Take ownership of columns by moving out of each unique pointer
        for (auto &col: views) {
            table_views.push_back(std::move(*col->inner));
        }

        auto table = std::make_unique<Column>();
        table->inner = cudf::concatenate(table_views);
        return table;
    }

    std::unique_ptr<Column> concat_column_views_on(
        rust::Slice<const std::unique_ptr<ColumnView>> views,
        const CudaStream &stream) {
        std::vector<cudf::column_view> column_views;
        column_views.reserve(views.size());
        for (auto &col: views) {
            column_views.push_back(std::move(*col->inner));
        }
        auto column = std::make_unique<Column>();
        column->inner = cudf::concatenate(column_views, stream.view());
        return column;
    }

    // Direct cuDF operations - 1:1 mappings
    std::unique_ptr<Table> apply_boolean_mask(const TableView &table, const ColumnView &boolean_mask) {
        auto result = std::make_unique<Table>();
        result->inner = cudf::apply_boolean_mask(*table.inner, *boolean_mask.inner);
        return result;
    }

    // Gather rows from a table based on a gather map
    std::unique_ptr<Table> gather(const TableView &source_table, const ColumnView &gather_map) {
        auto result = std::make_unique<Table>();
        result->inner = cudf::gather(
            *source_table.inner,
            *gather_map.inner,
            cudf::out_of_bounds_policy::DONT_CHECK
        );
        return result;
    }

    // Create a sliced view of a column
    std::unique_ptr<ColumnView> slice_column(const ColumnView &column, size_t offset, size_t length) {
        if (!column.inner) {
            throw std::runtime_error("Cannot slice null column view");
        }
        if (offset + length > static_cast<size_t>(column.inner->size())) {
            throw std::out_of_range("Slice bounds out of range");
        }

        // Use cuDF's native slice function from cudf/copying.hpp
        // slice() takes pairs of [start, end) indices and returns a vector of views
        auto start = static_cast<cudf::size_type>(offset);
        auto end = static_cast<cudf::size_type>(offset + length);
        std::vector indices = {start, end};

        auto sliced_views = cudf::slice(*column.inner, indices);

        // We expect exactly one view back since we provided one [start, end) pair
        if (sliced_views.empty()) {
            throw std::runtime_error("cudf::slice returned no views");
        }

        auto result = std::make_unique<ColumnView>();
        result->inner = std::make_unique<cudf::column_view>(sliced_views.at(0));
        return result;
    }

    rust::String get_cudf_version() {
        std::ostringstream version;
        version << CUDF_VERSION_MAJOR << "."
                << CUDF_VERSION_MINOR << "."
                << CUDF_VERSION_PATCH;
        return {version.str()};
    }

    // Per-stream wrapper around `cuda_async_memory_resource`.
    //
    // A single shared `cuda_async_memory_resource` becomes a hot spot once
    // multiple CUDA streams allocate from it concurrently: the CUDA driver
    // inserts cross-stream ordering events on every reuse, and per-call
    // `cudaMallocFromPoolAsync` cost grows from ~10 µs to ~36 µs in our
    // 4-stream agg workload. Each (segment, partition) in our pipeline is
    // bound to exactly one stream — there's no cross-stream traffic to
    // preserve — so we can give each stream its own CUDA mempool. The
    // driver then sees one stream per pool and elides the cross-stream
    // bookkeeping.
    class per_stream_async_mr final : public rmm::mr::device_memory_resource {
    public:
        // `release_threshold` is the cuda_async pool's release threshold —
        // memory cached up to this amount per pool stays resident on free,
        // anything above is returned to CUDA. We do *not* pre-reserve any
        // memory per pool; pools grow lazily via `cudaMallocFromPoolAsync`.
        // Pre-reserving N × initial up front would OOM for typical N=4.
        explicit per_stream_async_mr(std::optional<std::size_t> release_threshold)
            : release_threshold_{release_threshold} {}

    private:
        void* do_allocate(std::size_t bytes,
                          rmm::cuda_stream_view stream) override {
            return resource_for(stream)->allocate(bytes, stream);
        }

        void do_deallocate(void* p, std::size_t bytes,
                           rmm::cuda_stream_view stream) override {
            // `cudaFreeAsync` routes by ptr internally; any per-stream MR
            // works. We pick the one for this stream so RMM accounting
            // stays consistent.
            resource_for(stream)->deallocate(p, bytes, stream);
        }

        rmm::mr::cuda_async_memory_resource* resource_for(
            rmm::cuda_stream_view stream)
        {
            cudaStream_t key = stream.value();
            {
                std::shared_lock<std::shared_mutex> lock(mtx_);
                auto it = pools_.find(key);
                if (it != pools_.end()) return it->second.get();
            }
            std::unique_lock<std::shared_mutex> lock(mtx_);
            auto& slot = pools_[key];
            if (!slot) {
                // initial_pool_size = small explicit value (1 MiB). RMM's
                // default-when-nullopt is `free_device_memory / 2`, which
                // would have each per-stream pool grab half the GPU at
                // construction and OOM after a few streams.
                constexpr std::size_t small_initial = 1UL << 20;
                slot = std::make_unique<rmm::mr::cuda_async_memory_resource>(
                    std::optional<std::size_t>{small_initial},
                    release_threshold_);
            }
            return slot.get();
        }

    public:
        // Drop the pool associated with `stream`, returning any cached
        // memory to the OS via `cudaMemPoolDestroy`. Must be called before
        // the stream itself is destroyed.
        void release_for(cudaStream_t stream)
        {
            std::unique_lock<std::shared_mutex> lock(mtx_);
            pools_.erase(stream);
        }

        std::optional<std::size_t> release_threshold_;
        std::shared_mutex mtx_;
        std::unordered_map<
            cudaStream_t,
            std::unique_ptr<rmm::mr::cuda_async_memory_resource>>
            pools_;
    };

    namespace {
        // File-scope so `release_device_pool_stream` can reach it.
        std::unique_ptr<per_stream_async_mr> g_per_stream_mr;
    }

    bool config_device_memory_pool(size_t /*initial_bytes*/, size_t max_bytes) {
        // `initial_bytes` is intentionally ignored: pre-reserving N × initial
        // memory across N per-stream pools OOMs at typical configurations.
        // Pools grow lazily; `max_bytes` is used as the per-pool release
        // threshold (memory cached up to this size on free).
        if (g_per_stream_mr) return false;
        g_per_stream_mr = std::make_unique<per_stream_async_mr>(
            std::optional<std::size_t>{max_bytes});
        rmm::mr::set_current_device_resource(g_per_stream_mr.get());
        return true;
    }

    void release_device_pool_stream(const CudaStream& stream) {
        if (!g_per_stream_mr || !stream.is_valid()) return;
        g_per_stream_mr->release_for(stream.view().value());
    }

    bool config_pinned_memory_resource(size_t pool_size_bytes) {
        return cudf::config_default_pinned_memory_resource({.pool_size = pool_size_bytes});
    }

    void set_host_pinned_threshold(size_t threshold_bytes) {
        cudf::set_allocate_host_as_pinned_threshold(threshold_bytes);
    }

    // Arrow interop - convert Arrow data to cuDF table
    std::unique_ptr<Table> table_from_arrow_host(uint8_t const *schema_ptr, uint8_t const *device_array_ptr) {
        auto *schema = reinterpret_cast<const ArrowSchema *>(schema_ptr);
        auto *device_array = reinterpret_cast<const ArrowDeviceArray *>(device_array_ptr);

        auto result = std::make_unique<Table>();
        result->inner = cudf::from_arrow_host(schema, device_array);
        return result;
    }

    std::unique_ptr<Table> table_from_arrow_host_on(
        uint8_t const *schema_ptr,
        uint8_t const *device_array_ptr,
        const CudaStream &stream) {
        auto *schema = reinterpret_cast<const ArrowSchema *>(schema_ptr);
        auto *device_array = reinterpret_cast<const ArrowDeviceArray *>(device_array_ptr);
        auto result = std::make_unique<Table>();
        result->inner = cudf::from_arrow_host(schema, device_array, stream.view());
        return result;
    }

    // Arrow interop - convert Arrow array to cuDF column
    std::unique_ptr<Column> column_from_arrow(uint8_t const *schema_ptr, uint8_t const *array_ptr) {
        auto *schema = reinterpret_cast<const ArrowSchema *>(schema_ptr);
        auto *array = reinterpret_cast<const ArrowArray *>(array_ptr);

        auto result = std::make_unique<Column>();
        result->inner = cudf::from_arrow_column(schema, array);
        return result;
    }

    std::unique_ptr<Column> column_from_arrow_on(
        uint8_t const *schema_ptr,
        uint8_t const *array_ptr,
        const CudaStream &stream) {
        auto *schema = reinterpret_cast<const ArrowSchema *>(schema_ptr);
        auto *array = reinterpret_cast<const ArrowArray *>(array_ptr);
        auto result = std::make_unique<Column>();
        result->inner = cudf::from_arrow_column(schema, array, stream.view());
        return result;
    }
} // namespace libcudf_bridge
