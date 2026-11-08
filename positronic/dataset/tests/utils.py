from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from botocore.exceptions import ClientError

from positronic.dataset.episode import Episode, EpisodeContainer
from positronic.dataset.signal import IndicesLike, Signal
from positronic.dataset.time import TimeArray
from positronic.dataset.transforms import Elementwise, EpisodeTransform
from positronic.dataset.vector import RECORDED_TIME


class DummySignal(Signal[Any]):
    """Minimal array-backed Signal implementing the abstract API only.

    Used to validate core Signal's generic indexing/time logic and views.
    """

    def __init__(self, timestamps, values, *, timelines=(RECORDED_TIME,)):
        self._ts = TimeArray(timelines, np.asarray(timestamps, dtype=np.int64).reshape(len(timestamps), len(timelines)))
        self._ts.validate_order()
        self._vals = np.asarray(values)
        assert len(self._vals) == len(self._ts)

    @property
    def timelines(self):
        return self._ts.timelines

    def __len__(self):
        return len(self._ts)

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]):
        return self._ts.select(timelines).take(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[Any]:
        if not isinstance(indices, slice):
            indices = np.asarray(indices, dtype=np.int64)
        return cast(Sequence[Any], self._vals[indices])


class DummyTransform(EpisodeTransform):
    """Configurable transform for testing.

    Applies elementwise operations to signals from the input episode.

    Example:
        # Transform that multiplies 's' by 10 and outputs as 'a', and adds 1 to 's'
        DummyTransform(
            operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)},
            pass_through=True
        )
    """

    def __init__(self, operations: dict[str, tuple[str, Callable]], pass_through: bool | list[str] = False):
        """
        Args:
            operations: Dict mapping output_key -> (input_key, transform_func).
                       The transform_func receives arrays and returns transformed arrays.
            pass_through: Whether to pass through keys from input episode (True/False/list of keys)
        """
        self._operations = operations
        self._pass_through = pass_through

    def __call__(self, episode: Episode) -> Episode:
        data = {}

        # Apply all operations
        for out_key, (in_key, func) in self._operations.items():
            input_signal = episode[in_key]
            data[out_key] = Elementwise(input_signal, lambda seq, f=func: np.asarray(f(seq)))

        # Handle pass_through logic
        if self._pass_through is True:
            for key in episode:
                if key not in data:
                    data[key] = episode[key]
        elif isinstance(self._pass_through, list):
            for key in self._pass_through:
                if key not in data and key in episode:
                    data[key] = episode[key]

        return EpisodeContainer(data=data, meta=episode.meta)


class InMemoryS3:
    """The part of a boto3 S3 client that `pos3` calls, over a dict of `(bucket, key) -> bytes`."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        return {'ContentLength': len(self.objects[Bucket, Key])}

    def get_paginator(self, operation):
        assert operation == 'list_objects_v2'
        return self

    def paginate(self, *, Bucket, Prefix):
        yield {
            'Contents': [
                {'Key': key, 'Size': len(body)}
                for (bucket, key), body in sorted(self.objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def put_object(self, *, Bucket, Key, Body):
        self.objects[Bucket, Key] = Body

    def upload_file(self, filename, bucket, key, *, Callback):
        body = Path(filename).read_bytes()
        self.objects[bucket, key] = body
        Callback(len(body))

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def download_file(self, bucket, key, filename, *, Callback):
        body = self.objects[bucket, key]
        Path(filename).write_bytes(body)
        Callback(len(body))
