"""FastAPI server for exposing datasets over HTTP."""

from __future__ import annotations

import logging

import configuronic as cfn
import numpy as np
import pos3
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import positronic.cfg.ds
from positronic.dataset import Dataset
from positronic.dataset.signal import SupportsEncodedRepresentation
from positronic.utils.logging import init_logging
from positronic.utils.serialization import serialize

_dataset: Dataset | None = None
_app = FastAPI(title='Positronic Dataset Server', version='1.0.0')


class IndicesRequest(BaseModel):
    indices: list[int] | None = None
    slice: list[int | None] | None = None


class TimestampsRequest(BaseModel):
    timestamps: list[int]


@_app.get('/api/v1/dataset/info')
def dataset_info():
    ds = _get_dataset()
    return {'num_episodes': len(ds), 'meta': ds.meta}


@_app.get('/api/v1/episodes/{index}/info')
def episode_info(index: int):
    ep = _get_episode(index)
    signals_meta = {}
    for name, sig in ep.signals.items():
        supports_encoded = isinstance(sig, SupportsEncodedRepresentation)
        signals_meta[name] = {
            'length': len(sig),
            'kind': sig.kind.value,
            'dtype': np.dtype(sig.dtype).str,
            'shape': list(sig.shape) if sig.shape else [],
            'encoding_format': sig.encoding_format if supports_encoded else None,
        }
    return {'meta': ep.meta, 'static': ep.static, 'signals': signals_meta}


@_app.post('/api/v1/episodes/{ep}/signals/{sig}/timestamps')
def signal_timestamps(ep: int, sig: str, req: IndicesRequest):
    signal = _get_signal(ep, sig)
    indices = _parse_indices(req)
    return {'timestamps': np.asarray(signal._ts_at(indices)).tolist()}


@_app.post('/api/v1/episodes/{ep}/signals/{sig}/values')
def signal_values(ep: int, sig: str, req: IndicesRequest):
    signal = _get_signal(ep, sig)
    values = list(signal._values_at(_parse_indices(req)))
    return StreamingResponse(iter([serialize(values)]), media_type='application/msgpack')


@_app.post('/api/v1/episodes/{ep}/signals/{sig}/search')
def signal_search(ep: int, sig: str, req: TimestampsRequest):
    signal = _get_signal(ep, sig)
    indices = signal._search_ts(np.array(req.timestamps, dtype=np.int64))
    return {'indices': np.asarray(indices).tolist()}


@_app.get('/api/v1/episodes/{ep}/signals/{sig}/encoded')
def signal_encoded(ep: int, sig: str):
    signal = _get_signal(ep, sig)
    if not isinstance(signal, SupportsEncodedRepresentation):
        raise HTTPException(400, f'Signal {sig} does not support encoded representation')
    return StreamingResponse(
        signal.iter_encoded_chunks(),
        media_type='application/octet-stream',
        headers={'X-Encoding-Format': signal.encoding_format},
    )


@_app.post('/api/v1/episodes/{ep}/sample')
def episode_sample(ep: int, req: TimestampsRequest):
    episode = _get_episode(ep)
    timestamps = np.array(req.timestamps, dtype=np.int64)
    sampled = episode.time[timestamps]

    result_static, result_signals = {}, {}
    for key, value in sampled.items():
        if key in episode.signals:
            result_signals[key] = {'timestamps': timestamps.tolist(), 'values': serialize(list(value)).hex()}
        else:
            result_static[key] = value
    return {'static': result_static, 'signals': result_signals}


def _get_dataset() -> Dataset:
    if _dataset is None:
        raise HTTPException(503, 'Dataset not loaded')
    return _dataset


def _get_episode(index: int):
    try:
        return _get_dataset()[index]
    except IndexError as e:
        raise HTTPException(404, f'Episode {index} not found') from e


def _get_signal(ep: int, sig: str):
    signal = _get_episode(ep).signals.get(sig)
    if signal is None:
        raise HTTPException(404, f'Signal {sig} not found')
    return signal


def _parse_indices(req: IndicesRequest) -> slice | np.ndarray:
    if req.slice is not None:
        return slice(*req.slice)
    if req.indices is not None:
        return np.array(req.indices, dtype=np.int64)
    return slice(None)


@cfn.config(dataset=positronic.cfg.ds.local_all, host='0.0.0.0', port=8080, debug=False)
def main(dataset: Dataset, host: str, port: int, debug: bool):
    """Start the dataset server."""
    global _dataset
    _dataset = dataset
    logging.info(f'Starting dataset server with {len(dataset)} episodes at http://{host}:{port}')
    uvicorn.run(_app, host=host, port=port, log_level='debug' if debug else 'info')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
