"""Remote dataset client for accessing datasets over HTTP."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

import httpx
import numpy as np

from positronic.utils.serialization import deserialize

from .dataset import Dataset
from .episode import Episode, _EpisodeTimeIndexer
from .signal import TIMELINE_KEY, IndicesLike, Kind, RealNumericArrayLike, Signal, SignalMeta, validate_timeline

DATASET_API_PREFIX = '/api/v2'

T = TypeVar('T')


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
        r = self.session.get(f'{DATASET_API_PREFIX}/dataset/info')
        r.raise_for_status()
        return r.json()

    def get_episode_info(self, index: int) -> dict:
        r = self.session.get(f'{DATASET_API_PREFIX}/episodes/{index}/info')
        r.raise_for_status()
        info = r.json()
        info['static'] = deserialize(bytes.fromhex(info['static']))
        return info

    def get_signal_timestamps(self, ep: int, sig: str, indices: IndicesLike, *, timeline: str) -> np.ndarray:
        r = self.session.post(
            f'{DATASET_API_PREFIX}/episodes/{ep}/signals/{sig}/timestamps',
            json={**_encode_indices(indices), TIMELINE_KEY: timeline},
        )
        r.raise_for_status()
        return np.array(r.json()['timestamps'], dtype=np.int64)

    def get_signal_values(self, ep: int, sig: str, indices: IndicesLike) -> list:
        r = self.session.post(
            f'{DATASET_API_PREFIX}/episodes/{ep}/signals/{sig}/values',
            json=_encode_indices(indices),
            headers={'Accept': 'application/msgpack'},
        )
        r.raise_for_status()
        return deserialize(r.content)

    def search_signal_timestamps(
        self, ep: int, sig: str, ts_array: RealNumericArrayLike, *, timeline: str
    ) -> np.ndarray:
        r = self.session.post(
            f'{DATASET_API_PREFIX}/episodes/{ep}/signals/{sig}/search',
            json={'timestamps': np.asarray(ts_array).tolist(), TIMELINE_KEY: timeline},
        )
        r.raise_for_status()
        return np.array(r.json()['indices'], dtype=np.int64)

    def sample_episode(self, ep: int, timestamps: np.ndarray, *, timeline: str) -> dict:
        """Batch sample all signals at given timestamps."""
        r = self.session.post(
            f'{DATASET_API_PREFIX}/episodes/{ep}/sample',
            json={'timestamps': timestamps.tolist(), TIMELINE_KEY: timeline},
        )
        r.raise_for_status()
        data = r.json()
        result = deserialize(bytes.fromhex(data['static']))
        for sig_name, sig_data in data['signals'].items():
            result[sig_name] = deserialize(bytes.fromhex(sig_data['values']))
        return result

    def stream_encoded(self, ep: int, sig: str) -> Iterator[bytes]:
        with self.session.stream('GET', f'{DATASET_API_PREFIX}/episodes/{ep}/signals/{sig}/encoded') as r:
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
        meta: SignalMeta,
        length: int,
        encoding_format: str | None,
        timeline: str,
    ):
        validate_timeline(timeline)
        self._timeline = timeline
        self._client = client
        self._episode_index = episode_index
        self._signal_name = signal_name
        self._meta_cached = meta
        self._length = length
        self._encoding_format = encoding_format

    @property
    def timeline(self) -> str:
        return self._timeline

    def __len__(self) -> int:
        return self._length

    @property
    def meta(self) -> SignalMeta:
        return self._meta_cached

    def _ts_at(self, indices: IndicesLike, *, timeline: str) -> np.ndarray:
        self._check_timeline(timeline)
        return self._client.get_signal_timestamps(self._episode_index, self._signal_name, indices, timeline=timeline)

    def _values_at(self, indices: IndicesLike) -> Sequence[T]:
        return self._client.get_signal_values(self._episode_index, self._signal_name, indices)

    def _search_ts(self, ts_array: RealNumericArrayLike, *, timeline: str) -> np.ndarray:
        self._check_timeline(timeline)
        return self._client.search_signal_timestamps(
            self._episode_index, self._signal_name, ts_array, timeline=timeline
        )

    @property
    def encoding_format(self) -> str | None:
        return self._encoding_format

    def iter_encoded_chunks(self) -> Iterator[bytes]:
        if self._encoding_format is None:
            raise NotImplementedError("Signal doesn't support encoded representation")
        return self._client.stream_encoded(self._episode_index, self._signal_name)


class _RemoteEpisodeTimeIndexer:
    """Optimized time indexer using batch API for array access."""

    def __init__(self, episode: RemoteEpisode, timeline: str):
        validate_timeline(timeline)
        self._timeline = timeline
        self._episode = episode

    def __getitem__(self, timestamps):
        if isinstance(timestamps, np.ndarray):
            return self._episode._client.sample_episode(self._episode._index, timestamps, timeline=self._timeline)
        return _EpisodeTimeIndexer(self._episode, self._timeline)[timestamps]


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
                sig_meta = SignalMeta(
                    dtype=np.dtype(sig_info['dtype']),
                    shape=tuple(sig_info['shape']) if sig_info['shape'] else (),
                    kind=Kind(sig_info['kind']),
                )
                self._signals[name] = RemoteSignal(
                    self._client,
                    self._index,
                    name,
                    sig_meta,
                    sig_info['length'],
                    sig_info.get('encoding_format'),
                    sig_info[TIMELINE_KEY],
                )
            return self._signals[name]
        raise KeyError(f"'{name}' not found in episode {self._index}")

    @property
    def meta(self) -> dict:
        return dict(self._ensure_info()['meta'])

    def time(self, timeline: str):
        return _RemoteEpisodeTimeIndexer(self, timeline)


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
