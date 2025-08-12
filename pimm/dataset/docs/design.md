# Positronic Dataset library

This is a library for recording, storing, sharing and using robotic datasets. We differentiate between recording and storing the data and using it from with PyTorch when training. The first part is represented by data model while the second one is the view of that model.

## Core concepts
* __Stream__ – sequence of typed data with every element of this sequence being associated with one or more timestamps. Within one stream all the elements must have the same size (shape) and dtype, and we are checking for it. Three types are currently supported
  * __scalar__ of any supported type
  * __vector__ – of any length
  * __image__ – 3-channel images (of uint8 dtype)

We optimize for:
* Fast append during recording (low latency).
* Random access at query time by the timestamp.
* Window slices like "5 seconds before time X”.

## Public API
In the current version we have only "scalar" and "vector" streams supported. Also, we support one and only one timestamp type per stream. There are two main classes in the library.
```python
class Stream[T]:
    # Returns the value and timestamp of the closest record at or before the given timestamp, or None if not found
    def at(self, ts: int) -> Tuple[T, int] | None:
        pass
    # All the records in [start_ts, end_ts]
    def window(self, start_ts: int, end_ts: int) -> Tuple[Sequence[T], Sequence[int]]:
        pass

class StreamWriter[T]:
    # Appends data. Fails if `ts` is not increasing or the data is not compliant
    def append(self, data: T, ts: int) -> None:
        pass

    # Finalises the writing. All following `append` calls will fail
    def finish(self) -> None:
        pass
```

## Implementations

`Stream` and `StreamWriter` are abstract interfaces.

Currently we have only one scalar and vector streams implemented (`SimpleStream` and `SimpleStreamWriter`). These classes cover both scalars and vectors, and work for all fixed size data types supported by pyarrow.

The stream and writers for image data types (i.e. video) will be implemented later.

### Scalar / Vector

Every sequence is stored in one parquet file, with 'timestamp' and 'value' columns. The `at` and `window` methods rely on the binary search and hence have O(log N) compute time.

All the classes are lazy, in the sense that they don't perform any IO or computations until requested. The `SimpleStream` keeps all the data in pyarrow array in memory. Once the data is loaded into memory, we never make any copies of it, and we just provide the read-only access to the underlying data to the user.

We use `polars.search_sorted` to perform binary search operations.

