"""Remote dataset client for accessing datasets over HTTP."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Any, TypeVar

import httpx
import numpy as np

from positronic.utils.serialization import deserialize

from .dataset import Dataset
from .episode import Episode, _EpisodeTimeIndexer
from .signal import TIMELINES_KEY, IndicesLike, Kind, Signal, SignalMeta
from .time import Time, TimeArray, TimeBounds, validate_queries

T = TypeVar('T')
API_PREFIX = '/api/v2'
TIMESTAMP_VALUES_KEY = 'timestamps'
SIGNAL_BOUNDS_KEY = 'bounds'
SIGNAL_KIND_KEY = 'kind'
SIGNAL_DTYPE_KEY = 'dtype'
SIGNAL_SHAPE_KEY = 'shape'


def encode_times(times: Sequence[Time], *, timelines: tuple[str, ...]) -> dict:
    return {TIMELINES_KEY: timelines, TIMESTAMP_VALUES_KEY: [[time[name] for name in timelines] for time in times]}


def decode_times(data: dict) -> TimeArray:
    names = tuple(data[TIMELINES_KEY])
    rows = data[TIMESTAMP_VALUES_KEY]
    return TimeArray(names, np.array(rows, dtype=np.int64).reshape(len(rows), len(names)))


class DatasetClient:
    """HTTP client for dataset server communication."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout
        self._session: httpx.Client | None = None

    @property
    def session(self) -> httpx.Client:
        if self._session is None:
            self._session = httpx.Client(base_url=self._base_url, timeout=self._timeout)
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def get_dataset_info(self) -> dict:
        r = self.session.get(f'{API_PREFIX}/dataset/info')
        r.raise_for_status()
        return r.json()

    def get_episode_info(self, index: int) -> dict:
        r = self.session.get(f'{API_PREFIX}/episodes/{index}/info')
        r.raise_for_status()
        info = r.json()
        info['static'] = deserialize(bytes.fromhex(info['static']))
        return info

    def get_signal_timestamps(self, ep: int, sig: str, indices: IndicesLike, timelines: tuple[str, ...]) -> TimeArray:
        r = self.session.post(
            f'{API_PREFIX}/episodes/{ep}/signals/{sig}/timestamps',
            json={**_encode_indices(indices), TIMELINES_KEY: timelines},
        )
        r.raise_for_status()
        return decode_times(r.json())

    def get_signal_meta(self, ep: int, sig: str) -> SignalMeta:
        r = self.session.get(f'{API_PREFIX}/episodes/{ep}/signals/{sig}/meta')
        r.raise_for_status()
        data = r.json()
        return SignalMeta(
            dtype=np.dtype(data[SIGNAL_DTYPE_KEY]),
            shape=tuple(data[SIGNAL_SHAPE_KEY]),
            kind=Kind(data[SIGNAL_KIND_KEY]),
        )

    def get_signal_values(self, ep: int, sig: str, indices: IndicesLike) -> list:
        r = self.session.post(
            f'{API_PREFIX}/episodes/{ep}/signals/{sig}/values',
            json=_encode_indices(indices),
            headers={'Accept': 'application/msgpack'},
        )
        r.raise_for_status()
        return deserialize(r.content)

    def search_signal_timestamps(self, ep: int, sig: str, queries: Sequence[Time]) -> np.ndarray:
        payload = encode_times(queries, timelines=queries[0].timelines if len(queries) else ())
        r = self.session.post(f'{API_PREFIX}/episodes/{ep}/signals/{sig}/search', json=payload)
        r.raise_for_status()
        return np.array(r.json()['indices'], dtype=np.int64)

    def sample_episode(self, ep: int, timestamps: Sequence[Time]) -> dict:
        """Batch sample all signals at given timestamps."""
        payload = encode_times(timestamps, timelines=timestamps[0].timelines if len(timestamps) else ())
        r = self.session.post(f'{API_PREFIX}/episodes/{ep}/sample', json=payload)
        if r.status_code == 404:
            raise KeyError(r.json()['detail'])
        r.raise_for_status()
        data = r.json()
        result = deserialize(bytes.fromhex(data['static']))
        for sig_name, sig_data in data['signals'].items():
            result[sig_name] = deserialize(bytes.fromhex(sig_data['values']))
        return result

    def stream_encoded(self, ep: int, sig: str) -> Iterator[bytes]:
        with self.session.stream('GET', f'{API_PREFIX}/episodes/{ep}/signals/{sig}/encoded') as r:
            r.raise_for_status()
            yield from r.iter_bytes(chunk_size=64 * 1024)


def _encode_indices(indices: IndicesLike) -> dict:
    if isinstance(indices, slice):
        return {'slice': [indices.start, indices.stop, indices.step]}
    return {'indices': np.asarray(indices).tolist()}


class RemoteSignal(Signal[T]):
    """Signal backed by HTTP requests."""

    def __init__(
        self,
        client: DatasetClient,
        episode_index: int,
        signal_name: str,
        length: int,
        encoding_format: str | None,
        *,
        timelines: tuple[str, ...],
        bounds: Sequence[Time],
    ):
        self._client = client
        self._episode_index = episode_index
        self._signal_name = signal_name
        self._length = length
        self._encoding_format = encoding_format
        self._timelines = timelines
        self._time_bounds = bounds

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._timelines

    def __len__(self) -> int:
        return self._length

    @cached_property
    def meta(self) -> SignalMeta:
        if not len(self):
            raise ValueError('Signal is empty')
        return self._client.get_signal_meta(self._episode_index, self._signal_name)

    def _bounds(self, timelines: tuple[str, ...]) -> TimeBounds[Time]:
        return TimeBounds(self._time_bounds[0][timelines], self._time_bounds[1][timelines])

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        return self._client.get_signal_timestamps(self._episode_index, self._signal_name, indices, timelines)

    def _values_at(self, indices: IndicesLike) -> Sequence[T]:
        return self._client.get_signal_values(self._episode_index, self._signal_name, indices)

    def _search_ts(self, queries: Sequence[Time]) -> np.ndarray:
        return self._client.search_signal_timestamps(self._episode_index, self._signal_name, queries)

    @property
    def encoding_format(self) -> str | None:
        return self._encoding_format

    def iter_encoded_chunks(self) -> Iterator[bytes]:
        if self._encoding_format is None:
            raise NotImplementedError("Signal doesn't support encoded representation")
        return self._client.stream_encoded(self._episode_index, self._signal_name)


class _RemoteEpisodeTimeIndexer(_EpisodeTimeIndexer):
    def __init__(self, episode: RemoteEpisode):
        self.episode: RemoteEpisode = episode

    def __getitem__(self, request):
        if isinstance(request, Sequence) and not isinstance(request, str | bytes):
            validate_queries(request)
            return self.episode._client.sample_episode(self.episode._index, request)
        return super().__getitem__(request)


class RemoteEpisode(Episode):
    """Episode backed by HTTP requests."""

    def __init__(self, client: DatasetClient, index: int):
        self._client = client
        self._index = index
        self._info: dict | None = None
        self._signals: dict[str, RemoteSignal] = {}

    def _ensure_info(self) -> dict:
        if self._info is None:
            self._info = self._client.get_episode_info(self._index)
        return self._info

    def __iter__(self) -> Iterator[str]:
        info = self._ensure_info()
        yield from info['signals'].keys()
        yield from info['static'].keys()

    def __len__(self) -> int:
        info = self._ensure_info()
        return len(info['signals']) + len(info['static'])

    def __getitem__(self, name: str) -> Signal | Any:
        info = self._ensure_info()
        if name in info['static']:
            return info['static'][name]
        if name in info['signals']:
            if name not in self._signals:
                sig_info = info['signals'][name]
                self._signals[name] = RemoteSignal(
                    self._client,
                    self._index,
                    name,
                    sig_info['length'],
                    sig_info.get('encoding_format'),
                    timelines=tuple(sig_info[TIMELINES_KEY]),
                    bounds=tuple(decode_times(sig_info[SIGNAL_BOUNDS_KEY])),
                )
            return self._signals[name]
        raise KeyError(f"'{name}' not found in episode {self._index}")

    @property
    def meta(self) -> dict:
        return dict(self._ensure_info()['meta'])

    @property
    def time(self):
        return _RemoteEpisodeTimeIndexer(self)


class RemoteDataset(Dataset):
    """Dataset backed by HTTP requests to a remote server."""

    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self._client = DatasetClient(base_url, timeout=timeout)
        self._info: dict | None = None

    def _ensure_info(self) -> dict:
        if self._info is None:
            self._info = self._client.get_dataset_info()
        return self._info

    def __len__(self) -> int:
        return self._ensure_info()['num_episodes']

    def _get_episode(self, index: int) -> RemoteEpisode:
        return RemoteEpisode(self._client, index)

    @property
    def meta(self) -> dict:
        return self._ensure_info().get('meta', {})

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
