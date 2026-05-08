# `from_arrow_host` Pinned Buffer Lifetime Trace

You are right that the columns get `Arc` cloned. But those clones only live
inside `from_arrow_host`; they do not get stored in the returned `CuDFTable`.

Literal lifetime trace:

```rust
pub fn from_arrow_host(batch: RecordBatch) -> Result<Self, CuDFError> {
```

`batch` is moved into the function. It owns `Vec<ArrayRef>`, so it owns
`Arc<dyn Array>` references to the pinned arrays.

```rust
for col in batch.columns() { ... }
```

Borrow only. No lifetime change.

```rust
let schema = batch.schema().as_ref().clone();
```

Clones schema metadata only. No buffer ownership.

```rust
let struct_array = StructArray::from(batch);
```

This is the key Arrow code:

```rust
impl From<RecordBatch> for StructArray {
    fn from(value: RecordBatch) -> Self {
        Self {
            len: value.num_rows(),
            data_type: DataType::Struct(value.schema().fields().clone()),
            nulls: None,
            fields: value.columns().to_vec(),
        }
    }
}
```

Yes: `value.columns().to_vec()` clones the `ArrayRef`s. So the original
`RecordBatch` is consumed/dropped, but the pinned arrays stay alive through
`struct_array.fields`.

Then:

```rust
let array_data: ArrayData = struct_array.into_data();
```

For `StructArray`:

```rust
impl From<StructArray> for ArrayData {
    fn from(array: StructArray) -> Self {
        let builder = ArrayDataBuilder::new(array.data_type)
            .len(array.len)
            .nulls(array.nulls)
            .child_data(array.fields.iter().map(|x| x.to_data()).collect());

        unsafe { builder.build_unchecked() }
    }
}
```

This consumes `struct_array`. It calls `x.to_data()` on each child array. For
normal Arrow arrays, that `ArrayData` contains cloned `Buffer`s / `NullBuffer`s.
Those `Buffer`s hold `Arc<PinnedHostBuffer>` custom allocations. So now
`array_data` keeps the pinned buffers alive.

Then:

```rust
let ffi_array = FFI_ArrowArray::new(&array_data);
```

Arrow FFI clones the buffers again into FFI private data:

```rust
buffers: Vec<Option<Buffer>>,
```

So `ffi_array.private_data` also owns cloned `Buffer`s, which also keep the
pinned allocations alive.

Then:

```rust
let device_array = ArrowDeviceArray::new_cpu().with_array(ffi_array);
```

This moves `ffi_array` into `device_array.array`. So `device_array` owns the
Arrow C array and its private buffer refs.

Then:

```rust
let inner = unsafe {
    ffi::table_from_arrow_host(...)
}?;

Ok(Self { inner })
```

The C++ path calls cuDF:

```cpp
result->inner = cudf::from_arrow_host(schema, device_array, stream.inner, mr.inner);
```

And cuDF does async copies like:

```cpp
cudaMemcpyAsync(
    mutable_column_view.data<DeviceType>(),
    data_buffer + input->offset,
    sizeof(DeviceType) * num_rows,
    cudaMemcpyDefault,
    stream.value());
```

No stream sync there. It returns a `cudf::table` owning device buffers.

At the end of `from_arrow_host`, Rust drops the locals. That drops:

- `device_array`, which releases the FFI private data and its cloned buffers
- `array_data`, which releases its cloned buffers
- schema/stream/mr locals

So the pinned buffers are alive until the end of `from_arrow_host`, but not
after it returns.

That means the current code is safe only if cuDF's `from_arrow_host` completes
all reads from host memory before returning. But the local cuDF source shows it
enqueues `cudaMemcpyAsync` and returns the table. For pinned host memory, that
copy can still be in flight after return.

So the precise correction is:

- You are right that the pinned `RecordBatch`'s columns are `Arc` cloned.
- Those clones extend the pinned buffer lifetime through the FFI call.
- They do not extend it past `CuDFTable::from_arrow_host` returning.
- The returned `CuDFTable` does not own those pinned host buffers; it owns only
  the destination device columns.

That is why a sync after the current consuming `from_arrow_host(pinned_batch)`
is suspicious: by the time the caller can sync, the pinned host owners may
already have been dropped. The correct shape is still to keep the pinned batch
or equivalent buffer-owning `ArrayData`/FFI owner alive outside the call, then
sync/event/reaper it after the async copies are known complete.
