# Positronic Dataset library

This is a library for recording, storing, sharing and using robotic datasets. We differentiate between recording and storing the data and using it from with PyTorch when training. The first part is represented by data model while the second one is the view of that model.

## Core concepts
* __Signal__ – sequence of typed data with every element of this sequence being associated with one or more timestamps. Within one Signal all the elements must have the same size (shape) and dtype, and we are checking for it. Three types are currently supported
  * __scalar__ of any supported type
  * __vector__ – of any length
  * __image__ – 3-channel images (of uint8 dtype)

We optimize for:
* Fast append during recording (low latency).
* Random access at query time by the timestamp.
* Window slices like "5 seconds before time X".

## Public API
In the current version we have only "scalar" and "vector" Signals supported. Also, we support one and only one timestamp type per Signal. There are two main classes in the library.
```python
class Signal[T]:
    # Returns the number of records in the signal
    def __len__(self) -> int:
        pass

    # Access data by index or slice
    # signal[idx] returns (value, timestamp_ns) tuple
    # signal[start:end] returns a Signal view of the slice
    def __getitem__(self, index_or_slice: int | slice) -> Tuple[T, int] | Signal[T]:
        pass

    # Timestamp-based indexer property for accessing data by time
    # signal.time[ts_ns] returns (value, timestamp_ns) for closest record at or before ts_ns
    # signal.time[start_ts:end_ts] returns Signal view for time window [start_ts, end_ts)
    # signal.time[start:end:step] returns Signal with sampled values at step intervals
    @property
    def time(self) -> Sequence[Tuple[T, int]]:
        pass

class SignalWriter[T]:
    # Appends data with timestamp. Fails if ts_ns is not increasing or data shape/dtype doesn't match
    def append(self, data: T, ts_ns: int) -> None:
        pass

    # Finalizes the writing. All following append calls will fail
    def finish(self) -> None:
        pass
```

## Implementations

`Signal` and `SignalWriter` are abstract interfaces.

Currently we have only one scalar and vector Signals implemented (`SimpleSignal` and `SimpleSignalWriter`). These classes cover both scalars and vectors, and work for all fixed size data types supported by pyarrow.

The Signal and writers for image data types (i.e. video) will be implemented later.

### Scalar / Vector

Every sequence is stored in one parquet file, with 'timestamp' and 'value' columns. The timestamp-based access via the `time` property relies on binary search and hence has O(log N) compute time.

All the classes are lazy, in the sense that they don't perform any IO or computations until requested. The `SimpleSignal` keeps all the data in numpy arrays in memory after loading from the parquet file. Once the data is loaded into memory, we provide efficient access through views.

When accessing data via slices (either index-based like `signal[0:100]` or time-based like `signal.time[start_ts:end_ts]`), the library returns Signal views that share the underlying data with the original Signal. These views have the same API as the original Signal and provide zero-copy access to the data.

We use `numpy.searchsorted` to perform binary search operations for efficient timestamp-based lookups.

