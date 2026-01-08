"""Remote dataset client for accessing datasets over HTTP."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

import httpx
import numpy as np

from positronic.utils.serialization import deserialize

from .dataset import Dataset
from .episode import Episode, _EpisodeTimeIndexer
from .signal import IndicesLike, Kind, RealNumericArrayLike, Signal, SignalMeta

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
        r = self.session.get('/api/v1/dataset/info')
        r.raise_for_status()
        return r.json()

    def get_episode_info(self, index: int) -> dict:
        r = self.session.get(f'/api/v1/episodes/{index}/info')
        r.raise_for_status()
        return r.json()

    def get_signal_timestamps(self, ep: int, sig: str, indices: IndicesLike) -> np.ndarray:
        r = self.session.post(f'/api/v1/episodes/{ep}/signals/{sig}/timestamps', json=_encode_indices(indices))
        r.raise_for_status()
        return np.array(r.json()['timestamps'], dtype=np.int64)

    def get_signal_values(self, ep: int, sig: str, indices: IndicesLike) -> list:
        r = self.session.post(
            f'/api/v1/episodes/{ep}/signals/{sig}/values',
            json=_encode_indices(indices),
            headers={'Accept': 'application/msgpack'},
        )
        r.raise_for_status()
        return deserialize(r.content)

    def search_signal_timestamps(self, ep: int, sig: str, ts_array: RealNumericArrayLike) -> np.ndarray:
        r = self.session.post(
            f'/api/v1/episodes/{ep}/signals/{sig}/search', json={'timestamps': np.asarray(ts_array).tolist()}
        )
        r.raise_for_status()
        return np.array(r.json()['indices'], dtype=np.int64)

    def sample_episode(self, ep: int, timestamps: np.ndarray) -> dict:
        """Batch sample all signals at given timestamps."""
        r = self.session.post(f'/api/v1/episodes/{ep}/sample', json={'timestamps': timestamps.tolist()})
        r.raise_for_status()
        data = r.json()
        result = dict(data['static'])
        for sig_name, sig_data in data['signals'].items():
            result[sig_name] = deserialize(bytes.fromhex(sig_data['values']))
        return result

    def stream_encoded(self, ep: int, sig: str) -> Iterator[bytes]:
        with self.session.stream('GET', f'/api/v1/episodes/{ep}/signals/{sig}/encoded') as r:
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
    ):
        self._client = client
        self._episode_index = episode_index
        self._signal_name = signal_name
        self._meta_cached = meta
        self._length = length
        self._encoding_format = encoding_format

    def __len__(self) -> int:
        return self._length

    @property
    def meta(self) -> SignalMeta:
        return self._meta_cached

    def _ts_at(self, indices: IndicesLike) -> np.ndarray:
        return self._client.get_signal_timestamps(self._episode_index, self._signal_name, indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[T]:
        return self._client.get_signal_values(self._episode_index, self._signal_name, indices)

    def _search_ts(self, ts_array: RealNumericArrayLike) -> np.ndarray:
        return self._client.search_signal_timestamps(self._episode_index, self._signal_name, ts_array)

    @property
    def encoding_format(self) -> str | None:
        return self._encoding_format

    def iter_encoded_chunks(self) -> Iterator[bytes]:
        if self._encoding_format is None:
            raise NotImplementedError("Signal doesn't support encoded representation")
        return self._client.stream_encoded(self._episode_index, self._signal_name)


class _RemoteEpisodeTimeIndexer:
    """Optimized time indexer using batch API for array access."""

    def __init__(self, episode: RemoteEpisode):
        self._episode = episode

    def __getitem__(self, timestamps):
        if isinstance(timestamps, np.ndarray):
            return self._episode._client.sample_episode(self._episode._index, timestamps)
        return _EpisodeTimeIndexer(self._episode)[timestamps]


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
                    self._client, self._index, name, sig_meta, sig_info['length'], sig_info.get('encoding_format')
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
