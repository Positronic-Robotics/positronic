"""Tests for remote dataset server and client."""

from __future__ import annotations

import threading
import time

import numpy as np
import pos3
import pytest
import uvicorn
from fastapi.testclient import TestClient

from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.remote import RemoteDataset
from positronic.dataset.remote_server import server as remote_server
from positronic.dataset.signal import SupportsEncodedRepresentation
from positronic.dataset.utilities.migrate_remote import migrate_remote_dataset
from positronic.dataset.video import VideoSignal, VideoSignalWriter
from positronic.utils.serialization import deserialize


@pytest.fixture
def dataset_with_video(tmp_path):
    """Create a dataset with video and numeric signals."""
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as w:
        for ep_idx in range(2):
            with w.new_episode() as ew:
                ew.set_static('task', f'task_{ep_idx}')
                ew.set_static('episode_id', ep_idx)

                # Numeric signal
                for i in range(5):
                    ew.append('action', np.array([i * 0.1, i * 0.2], dtype=np.float32), ts_ns=1000 + i * 100)

                # Video signal
                video_path = ew.path / 'cam.mp4'
                frames_path = ew.path / 'cam.frames.parquet'
                with VideoSignalWriter(video_path, frames_path, fps=30) as vw:
                    for i in range(3):
                        frame = np.full((64, 64, 3), (ep_idx + 1) * 50 + i * 10, dtype=np.uint8)
                        vw.append(frame, ts_ns=1000 + i * 100)

    return LocalDataset(root)


@pytest.fixture
def test_client(dataset_with_video):
    """Create a FastAPI TestClient with the dataset."""
    remote_server._dataset = dataset_with_video
    return TestClient(remote_server._app)


# --- Server endpoint tests ---


def test_dataset_info_endpoint(test_client, dataset_with_video):
    r = test_client.get('/api/v1/dataset/info')
    assert r.status_code == 200
    data = r.json()
    assert data['num_episodes'] == 2
    assert 'meta' in data


def test_episode_info_endpoint(test_client):
    r = test_client.get('/api/v1/episodes/0/info')
    assert r.status_code == 200
    data = r.json()
    assert data['static']['task'] == 'task_0'
    assert data['static']['episode_id'] == 0
    assert 'action' in data['signals']
    assert 'cam' in data['signals']
    assert data['signals']['action']['length'] == 5
    assert data['signals']['cam']['length'] == 3
    assert data['signals']['cam']['encoding_format'] == 'positronic.video.v1'


def test_episode_info_not_found(test_client):
    r = test_client.get('/api/v1/episodes/999/info')
    assert r.status_code == 404


def test_signal_timestamps_endpoint(test_client):
    # Test with indices
    r = test_client.post('/api/v1/episodes/0/signals/action/timestamps', json={'indices': [0, 2, 4]})
    assert r.status_code == 200
    data = r.json()
    assert data['timestamps'] == [1000, 1200, 1400]

    # Test with slice
    r = test_client.post('/api/v1/episodes/0/signals/action/timestamps', json={'slice': [0, 3, None]})
    assert r.status_code == 200
    data = r.json()
    assert data['timestamps'] == [1000, 1100, 1200]


def test_signal_values_endpoint(test_client):
    r = test_client.post('/api/v1/episodes/0/signals/action/values', json={'indices': [0, 1]})
    assert r.status_code == 200
    values = deserialize(r.content)
    assert len(values) == 2
    np.testing.assert_allclose(values[0], [0.0, 0.0])
    np.testing.assert_allclose(values[1], [0.1, 0.2])


def test_signal_search_endpoint(test_client):
    r = test_client.post('/api/v1/episodes/0/signals/action/search', json={'timestamps': [1050, 1150]})
    assert r.status_code == 200
    data = r.json()
    assert data['indices'] == [0, 1]


def test_signal_encoded_endpoint(test_client):
    r = test_client.get('/api/v1/episodes/0/signals/cam/encoded')
    assert r.status_code == 200
    assert r.headers['x-encoding-format'] == 'positronic.video.v1'
    assert len(r.content) > 0


def test_signal_encoded_not_supported(test_client):
    r = test_client.get('/api/v1/episodes/0/signals/action/encoded')
    assert r.status_code == 400


def test_episode_sample_endpoint(test_client):
    r = test_client.post('/api/v1/episodes/0/sample', json={'timestamps': [1000, 1100]})
    assert r.status_code == 200
    data = r.json()
    assert data['static']['task'] == 'task_0'
    assert 'action' in data['signals']
    action_values = deserialize(bytes.fromhex(data['signals']['action']['values']))
    assert len(action_values) == 2


# --- RemoteDataset client tests ---


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server(dataset_with_video):
    """Start a real server in a background thread."""
    port = find_free_port()
    remote_server._dataset = dataset_with_video

    config = uvicorn.Config(remote_server._app, host='127.0.0.1', port=port, log_level='error')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):
        try:
            import httpx

            httpx.get(f'http://127.0.0.1:{port}/api/v1/dataset/info', timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError('Server did not start')

    yield f'http://127.0.0.1:{port}'

    server.should_exit = True
    thread.join(timeout=2)


def test_remote_dataset_len(running_server):
    with RemoteDataset(running_server) as ds:
        assert len(ds) == 2


def test_remote_dataset_episode_static(running_server):
    with RemoteDataset(running_server) as ds:
        ep = ds[0]
        assert ep['task'] == 'task_0'
        assert ep['episode_id'] == 0


def test_remote_dataset_signal_access(running_server):
    with RemoteDataset(running_server) as ds:
        signal = ds[0]['action']
        assert len(signal) == 5
        value, ts = signal[0]
        np.testing.assert_allclose(value, [0.0, 0.0])
        assert ts == 1000


def test_remote_dataset_signal_slice(running_server):
    with RemoteDataset(running_server) as ds:
        signal = ds[0]['action']
        values = list(signal[1:3])
        assert len(values) == 2
        np.testing.assert_allclose(values[0][0], [0.1, 0.2])
        np.testing.assert_allclose(values[1][0], [0.2, 0.4])


def test_remote_dataset_time_indexer(running_server):
    with RemoteDataset(running_server) as ds:
        ep = ds[0]
        timestamps = np.array([1000, 1100], dtype=np.int64)
        result = ep.time[timestamps]
        assert 'task' in result
        assert 'action' in result
        assert len(result['action']) == 2


def test_remote_dataset_video_encoded_stream(running_server):
    with RemoteDataset(running_server) as ds:
        signal = ds[0]['cam']
        assert signal.encoding_format == 'positronic.video.v1'
        chunks = list(signal.iter_encoded_chunks())
        assert len(chunks) > 0
        total_size = sum(len(c) for c in chunks)
        assert total_size > 0


def test_remote_dataset_iteration(running_server):
    with RemoteDataset(running_server) as ds:
        episodes = list(ds)
        assert len(episodes) == 2
        assert episodes[0]['episode_id'] == 0
        assert episodes[1]['episode_id'] == 1


# --- Migration tests ---


def test_migrate_remote_dataset_numeric_only(tmp_path):
    """Test migration of dataset with numeric signals only."""
    source_root = tmp_path / 'source'
    dest_root = tmp_path / 'dest'

    with LocalDatasetWriter(source_root) as w:
        for i in range(2):
            with w.new_episode() as ew:
                ew.set_static('id', i)
                for j in range(3):
                    ew.append('signal', np.array([j], dtype=np.float32), ts_ns=1000 + j * 100)

    source_ds = LocalDataset(source_root)
    port = find_free_port()
    remote_server._dataset = source_ds

    config = uvicorn.Config(remote_server._app, host='127.0.0.1', port=port, log_level='error')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            import httpx

            httpx.get(f'http://127.0.0.1:{port}/api/v1/dataset/info', timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)

    try:
        with pos3.mirror():
            migrate_remote_dataset(f'http://127.0.0.1:{port}', str(dest_root))
    finally:
        server.should_exit = True
        thread.join(timeout=2)

    dest_ds = LocalDataset(dest_root)
    assert len(dest_ds) == 2
    assert dest_ds[0]['id'] == 0
    signal = dest_ds[0]['signal']
    assert len(signal) == 3
    np.testing.assert_allclose(signal[0][0], [0])


def test_migrate_remote_dataset_with_video(running_server, tmp_path):
    """Test migration preserves video without re-encoding."""
    dest_root = tmp_path / 'migrated'

    with pos3.mirror():
        migrate_remote_dataset(running_server, str(dest_root))

    dest_ds = LocalDataset(dest_root)
    assert len(dest_ds) == 2

    ep = dest_ds[0]
    assert ep['task'] == 'task_0'

    # Check video signal exists and is readable
    cam = ep['cam']
    assert isinstance(cam, VideoSignal)
    assert len(cam) == 3

    # Verify video can be decoded
    frame, ts = cam[0]
    assert frame.shape == (64, 64, 3)
    assert ts == 1000


def test_video_signal_supports_encoded_protocol(dataset_with_video):
    """Verify VideoSignal implements SupportsEncodedRepresentation."""
    ep = dataset_with_video[0]
    cam = ep['cam']
    assert isinstance(cam, SupportsEncodedRepresentation)
    assert cam.encoding_format == 'positronic.video.v1'
    chunks = list(cam.iter_encoded_chunks())
    assert len(chunks) > 0
