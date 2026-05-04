# End-to-End Query Execution Flow in libcudf-rs

This document provides a comprehensive walkthrough of how a query executes from Rust → C++ → GPU and back, with detailed code references and diagrams.

## Query Example

```sql
SELECT c, COUNT(a) FROM table GROUP BY c
```

## 🗺️ Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATAFUSION QUERY PLAN                         │
│  SELECT c, COUNT(a) FROM table GROUP BY c                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. CuDFUnloadExec (GPU → CPU)                                   │
│    libcudf-datafusion/src/physical/cudf_unload.rs:76            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CuDFAggregateExec (GPU Aggregation)                          │
│    libcudf-datafusion/src/aggregate/mod.rs:132                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CuDFLoadExec (CPU → GPU)                                     │
│    libcudf-datafusion/src/physical/cudf_load.rs:72              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. TestMemoryExec (Data Source)                                 │
│    Input: RecordBatch[(a=Int64, c=Utf8)]                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Phase 1: Data Flows UP from Source

### Input Data (CPU Memory)

```
RecordBatch {
    a: [1, 4, 3]                    // Int64 column
    c: ["hello", "hello", "world"]  // String column
}
```

---

## 🚀 Phase 2: CPU → GPU Transfer (`CuDFLoadExec`)

**File:** `libcudf-datafusion/src/physical/cudf_load.rs:72-90`

```rust
fn execute(&self, partition: usize, context: Arc<TaskContext>)
    -> Result<SendableRecordBatchStream> {
    let host_stream = self.input.execute(partition, context)?;

    let cudf_stream = host_stream.map(move |batch_or_err| {
        let batch = batch_or_err?;

        // Convert each Arrow array to GPU column
        let mut gpu_arrays = Vec::with_capacity(batch.num_columns());
        for array in batch.columns() {
            // 🔥 THIS IS WHERE DATA GOES TO GPU
            let cudf_column = CuDFColumn::from_arrow_host(array.as_ref())
                .map_err(cudf_to_df)?;
            gpu_arrays.push(Arc::new(cudf_column.into_view()));
        }

        Ok(RecordBatch::try_new(target_schema.clone(), gpu_arrays)?)
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(target_schema, cudf_stream)))
}
```

### What happens inside `CuDFColumn::from_arrow_host`?

**File:** `src/column.rs` (simplified flow)

```rust
pub fn from_arrow_host(array: &dyn Array) -> Result<Self> {
    // 1. Convert Arrow array to ArrowArray FFI structure
    let ffi_array = arrow::ffi::FFI_ArrowArray::new(array.to_data());

    // 2. Call C++ to copy to GPU
    let cudf_column = libcudf_sys::ffi::column_from_arrow_host(
        &ffi_array as *const _ as *mut u8
    );

    Ok(Self::new(cudf_column))
}
```

**File:** `libcudf-sys/src/column.cpp:200-250` (C++ side)

```cpp
std::unique_ptr<Column> column_from_arrow_host(uint8_t *array_ptr) {
    auto *device_array = reinterpret_cast<ArrowDeviceArray*>(array_ptr);

    // 🔥 cuDF copies data from CPU to GPU here
    auto cudf_table = cudf::from_arrow_host(*device_array);

    auto column = std::make_unique<Column>();
    column->inner = std::move(cudf_table->release()[0]);
    return column;
}
```

### Memory State After Load

```
CPU (Host):                      GPU (Device):
─────────────                    ──────────────
a: [1, 4, 3]        ──────>      a: [1, 4, 3]
c: ["hello",...]    ──────>      c: ["hello",...]
```

---

## ⚙️ Phase 3: GPU Aggregation (`CuDFAggregateExec`)

**File:** `libcudf-datafusion/src/aggregate/mod.rs:132-145`

```rust
fn execute(&self, partition: usize, context: Arc<TaskContext>)
    -> Result<SendableRecordBatchStream> {
    let input = self.input.execute(partition, context)?;

    // Create streaming aggregation processor
    let stream = stream::Stream::new(
        input,                    // GPU data stream
        self.schema(),
        self.group_by.clone(),    // GROUP BY c
        self.aggr_expr.clone(),   // COUNT(a)
    );

    Ok(Box::pin(stream))
}
```

### Stream Processing Logic

**File:** `libcudf-datafusion/src/aggregate/stream.rs:146-231`

```rust
fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>)
    -> Poll<Option<Result<RecordBatch>>> {
    loop {
        match &self.state {
            State::ReceivingInput => {
                match ready!(self.input.poll_next_unpin(cx)) {
                    Some(Ok(batch)) => {
                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        // STEP 3A: Extract GROUP BY columns
                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        let group = evaluate_group_by(&self.group_by, &batch)?;
                        let column_views = group[0]
                            .into_iter()
                            .map(|x| x.as_any().downcast_ref::<CuDFColumnView>())
                            .cloned()
                            .collect::<Vec<_>>();

                        // Create groupby with keys table
                        let table_view = CuDFTableView::from_column_views(column_views)?;
                        let group_by = CuDFGroupBy::from_table_view(table_view);

                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        // STEP 3B: Build aggregation requests
                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        let mut requests = Vec::new();
                        for (agg_op, args) in self.aggregate_ops.iter().zip(evaluated_views) {
                            // For COUNT: create partial request
                            requests.extend(agg_op.partial_requests(&args)?);
                        }

                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        // STEP 3C: Execute on GPU! 🔥
                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        let (keys, results) = group_by.aggregate(&requests)?;

                        // Store partial results
                        self.results.push(results);
                        self.keys.push(keys);
                    }
                    None => {
                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        // STEP 3D: Final aggregation phase
                        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        self.state = State::Final;
                    }
                }
            }

            State::Final => {
                // Concatenate all partial results
                let keys_table = self.concat_keys()?;
                let concatenated_columns = self.concat_partial_results()?;

                // Final aggregation
                let group_by = CuDFGroupBy::from_table_view(keys_table.into_view());
                let mut requests = Vec::new();
                for (agg, args) in self.aggregate_ops.iter().zip(concatenated_columns.iter()) {
                    requests.extend(agg.final_requests(args)?);
                }

                // 🔥 Final GPU aggregation
                let (keys, results) = group_by.aggregate(&requests)?;
                let output = self.build_final_batch(keys, results)?;

                return Poll::Ready(Some(Ok(output)));
            }
        }
    }
}
```

### COUNT Aggregation Implementation

**File:** `libcudf-datafusion/src/aggregate/op/count.rs:23-62`

```rust
impl CuDFAggregationOp for CuDFCount {
    fn partial_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<AggregationRequest>> {
        // Partial: COUNT input values per batch
        let mut request = AggregationRequest::from_column_view(args[0].clone());
        request.add(AggregationOp::COUNT.group_by());
        Ok(vec![request])
    }

    fn final_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<AggregationRequest>> {
        // Final: SUM the partial counts
        let mut request = AggregationRequest::from_column_view(args[0].clone());
        request.add(AggregationOp::SUM.group_by());
        Ok(vec![request])
    }

    fn merge(&self, args: &[CuDFColumnView]) -> Result<CuDFColumnView> {
        // cuDF COUNT returns Int32, but DataFusion expects Int64
        let result_array = args[0].to_arrow_host()?;
        let casted = cast(&result_array, &DataType::Int64)?;
        let column = CuDFColumn::from_arrow_host(casted.as_ref())?;
        Ok(column.into_view())
    }
}
```

---

## 🔧 Phase 4: C++ Groupby Execution

**File:** `src/group_by.rs:41-69`

```rust
pub fn aggregate(&self, requests: &[AggregationRequest])
    -> Result<(CuDFTable, Vec<Vec<CuDFColumn>>)> {
    // Collect request pointers
    let requests = requests.iter().map(|x| x.inner.as_ptr()).collect::<Vec<_>>();

    // 🔥 Call into C++
    let mut gby_result = self.inner.aggregate(&requests)?;

    // Extract results
    let keys = gby_result.pin_mut().release_keys();
    let keys = CuDFTable::from_ptr(keys);

    let mut results = Vec::new();
    for i in 0..gby_result.len() {
        let mut released_result = gby_result.pin_mut().release_result(i);
        let mut cols = Vec::new();
        for j in 0..released_result.len() {
            let col = released_result.pin_mut().release(j);
            cols.push(CuDFColumn::new(col));
        }
        results.push(cols)
    }

    Ok((keys, results))
}
```

**File:** `libcudf-sys/src/groupby.cpp:42-71`

```cpp
std::unique_ptr<GroupByResult> GroupBy::aggregate(
    rust::Slice<const AggregationRequest * const> requests) const {

    // Convert Rust requests to cuDF requests
    std::vector<cudf::groupby::aggregation_request> cudf_requests;
    cudf_requests.reserve(requests.size());

    for (auto *req: requests) {
        cudf::groupby::aggregation_request cudf_req;
        cudf_req.values = req->inner->values;  // Column to aggregate

        // Clone aggregations (e.g., COUNT)
        for (auto &agg: req->inner->aggregations) {
            auto cloned = agg->clone();
            auto *groupby_agg = dynamic_cast<cudf::groupby_aggregation *>(cloned.release());
            cudf_req.aggregations.push_back(
                std::unique_ptr<cudf::groupby_aggregation>(groupby_agg)
            );
        }
        cudf_requests.push_back(std::move(cudf_req));
    }

    // 🔥🔥🔥 THIS LAUNCHES GPU KERNELS! 🔥🔥🔥
    auto aggregate_result = inner->aggregate(cudf_requests);
    //         ^^^^^^
    //         This is cuDF's groupby::groupby::aggregate()
    //         It launches CUDA kernels that execute on the GPU

    // Package results for Rust
    auto group_by_result = std::make_unique<GroupByResult>();
    group_by_result->keys.inner = std::move(aggregate_result.first);

    for (auto &cudf_agg_result: aggregate_result.second) {
        auto result = std::vector<Column>();
        for (auto &col: cudf_agg_result.results) {
            result.emplace_back(column_from_unique_ptr(std::move(col)));
        }
        group_by_result->results.emplace_back(std::move(result));
    }

    return group_by_result;
}
```

### GPU Kernel Execution (Inside cuDF)

```
┌─────────────────────────────────────────────────────────────┐
│ GPU MEMORY (CUDA Kernels)                                   │
│                                                              │
│ Input:                                                       │
│   Keys:   ["hello", "hello", "world"]                       │
│   Values: [1, 4, 3]                                          │
│                                                              │
│ Kernel 1: Hash grouping by "c" column                       │
│   Thread blocks process rows in parallel                    │
│   Result: {"hello": [1,4], "world": [3]}                    │
│                                                              │
│ Kernel 2: COUNT aggregation per group                       │
│   Each thread block counts elements in its group            │
│   Result: {"hello": 2, "world": 1}                          │
│                                                              │
│ Output (still in GPU memory):                               │
│   Keys:   ["world", "hello"]                                │
│   Counts: [1, 2]                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔙 Phase 5: GPU → CPU Transfer (`CuDFUnloadExec`)

**File:** `libcudf-datafusion/src/physical/cudf_unload.rs:76-95`

```rust
fn execute(&self, partition: usize, context: Arc<TaskContext>)
    -> Result<SendableRecordBatchStream> {
    let gpu_stream = self.input.execute(partition, context)?;

    let host_stream = gpu_stream.map(move |batch_or_err| {
        let batch = batch_or_err?;

        // Convert each GPU column back to CPU
        let mut host_arrays = Vec::with_capacity(batch.num_columns());
        for array in batch.columns() {
            let cudf_view = array
                .as_any()
                .downcast_ref::<CuDFColumnView>()
                .ok_or_else(|| exec_err!("Expected CuDFColumnView"))?;

            // 🔥 THIS IS WHERE DATA COMES FROM GPU
            let host_array = cudf_view.to_arrow_host().map_err(cudf_to_df)?;
            host_arrays.push(host_array);
        }

        Ok(RecordBatch::try_new(target_schema.clone(), host_arrays)?)
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(target_schema, host_stream)))
}
```

### What happens inside `to_arrow_host`?

**File:** `src/column_view.rs:80-110`

```rust
pub fn to_arrow_host(&self) -> Result<ArrayRef> {
    let mut device_array = libcudf_sys::ArrowDeviceArray::new_cpu();

    // 🔥 Call C++ to copy from GPU to CPU
    unsafe {
        let device_array_ptr = &mut device_array as *mut _ as *mut u8;
        self.inner.to_arrow_array(device_array_ptr);
    }

    // Convert from FFI to Arrow
    let array_data = unsafe {
        arrow::ffi::from_ffi_and_data_type(device_array.array, self.dt.clone())?
    };

    Ok(arrow::array::make_array(array_data))
}
```

**File:** `libcudf-sys/src/column.cpp:36-44`

```cpp
void ColumnView::to_arrow_array(uint8_t *out_array_ptr) const {
    if (!inner) {
        throw std::runtime_error("Cannot convert null column view");
    }

    // 🔥 cuDF copies data from GPU to CPU here
    auto device_array_unique = cudf::to_arrow_host(*this->inner);

    auto *out_array = reinterpret_cast<ArrowDeviceArray*>(out_array_ptr);
    *out_array = *device_array_unique.get();
    device_array_unique.release();
}
```

---

## 📤 Phase 6: Final Result

### Memory State After Unload

```
GPU (Device):                    CPU (Host):
──────────────                   ─────────────
Keys:   ["world", "hello"]  ──>  Keys:   ["world", "hello"]
Counts: [1, 2]              ──>  Counts: [1, 2]
```

### Final RecordBatch (CPU)

```
+-------+----------+
| c     | COUNT(a) |
+-------+----------+
| world | 1        |
| hello | 2        |
+-------+----------+
```

---

## 🔄 Complete Data Flow Diagram

```
                        RUST SPACE
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  INPUT: RecordBatch (CPU)                               │
│  ┌──────────────────────────┐                           │
│  │ a: [1, 4, 3]            │                           │
│  │ c: ["hello","hello"...] │                           │
│  └──────────────────────────┘                           │
│                │                                         │
│                ▼                                         │
│  CuDFLoadExec::execute()                                │
│  ├─> CuDFColumn::from_arrow_host()                      │
│  │                                                       │
└──┼───────────────────────────────────────────────────────┘
   │
   │   FFI BOUNDARY (cxx crate)
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│                      C++ SPACE                           │
│                                                          │
│  column_from_arrow_host()                               │
│  ├─> cudf::from_arrow_host()                            │
│  │        │                                              │
│  │        ▼                                              │
│  │   ┌─────────────────────────┐                        │
│  │   │ CUDA API: cudaMemcpy() │  ──────────────┐       │
│  │   └─────────────────────────┘                │       │
│  │                                               │       │
└──┼───────────────────────────────────────────────┼───────┘
   │                                               │
   ▼                                               ▼
┌─────────────────────────┐           ┌──────────────────────┐
│    CPU MEMORY           │           │    GPU MEMORY        │
│  ─────────────          │           │  ─────────────       │
│  a: [1, 4, 3]          │  ═════>   │  a: [1, 4, 3]       │
│  c: ["hello",...]      │           │  c: ["hello",...]   │
└─────────────────────────┘           └──────────────────────┘
                                                  │
                                                  ▼
                             ┌──────────────────────────────┐
                             │   GPU CUDA KERNELS           │
                             │   ─────────────────          │
                             │   1. Hash grouping           │
                             │   2. COUNT aggregation       │
                             │                              │
                             │   Result:                    │
                             │   Keys: ["world", "hello"]   │
                             │   Count: [1, 2]              │
                             └──────────────────────────────┘
                                                  │
   ┌──────────────────────────────────────────────┘
   │
   │   C++ SPACE
   ▼
┌──────────────────────────────────────────────────────────┐
│  GroupBy::aggregate() returns                            │
│  │                                                        │
│  └─> to_arrow_host()                                     │
│       └─> cudf::to_arrow_host()                          │
│            └─> cudaMemcpy() (GPU → CPU)                  │
│                                                          │
└──┬───────────────────────────────────────────────────────┘
   │
   │   FFI BOUNDARY
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│                      RUST SPACE                          │
│                                                          │
│  CuDFUnloadExec::execute()                              │
│  ├─> CuDFColumnView::to_arrow_host()                    │
│  │                                                       │
│  ▼                                                       │
│  OUTPUT: RecordBatch (CPU)                              │
│  ┌──────────────────────────┐                           │
│  │ c:     ["world","hello"] │                           │
│  │ COUNT: [1, 2]            │                           │
│  └──────────────────────────┘                           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Takeaways

1. **Two FFI Crossings:** Data crosses Rust ↔ C++ boundary twice (load and unload)
2. **Two Memory Transfers:** Data copied CPU → GPU (load), then GPU → CPU (unload)
3. **GPU Execution is Opaque:** cuDF handles all GPU kernel launches internally
4. **Streaming Design:** Processes batches incrementally with partial aggregations
5. **Type Safety:** Arrow's type system maintained throughout the pipeline
6. **Zero-Copy Views:** Within GPU, operations use views (no copying) until final result

---

## 📁 File Reference Index

### Rust Files (High-Level API)
- `libcudf-datafusion/src/aggregate/mod.rs` - Main aggregation executor
- `libcudf-datafusion/src/aggregate/stream.rs` - Streaming aggregation logic
- `libcudf-datafusion/src/aggregate/op/count.rs` - COUNT implementation
- `libcudf-datafusion/src/physical/cudf_load.rs` - CPU → GPU transfer
- `libcudf-datafusion/src/physical/cudf_unload.rs` - GPU → CPU transfer
- `src/group_by.rs` - GroupBy high-level API
- `src/column.rs` - Column high-level API
- `src/column_view.rs` - Column view implementation
- `src/scalar.rs` - Scalar type with Array trait

### C++ Files (FFI Layer)
- `libcudf-sys/src/groupby.cpp` - GroupBy C++ bridge
- `libcudf-sys/src/aggregation.cpp` - Aggregation factory functions
- `libcudf-sys/src/column.cpp` - Column FFI operations
- `libcudf-sys/src/lib.rs` - cxx bridge definitions

### Key Concepts
- **CuDFLoadExec**: Transfers Arrow RecordBatch from CPU to GPU
- **CuDFAggregateExec**: Executes aggregations on GPU data
- **CuDFUnloadExec**: Transfers results from GPU back to CPU
- **CuDFGroupBy**: Wraps cuDF's groupby functionality
- **AggregationRequest**: Specifies columns and aggregations to compute
- **Stream**: Async iterator that processes batches incrementally

---

This architecture allows DataFusion queries to transparently accelerate on GPU! 🚀
