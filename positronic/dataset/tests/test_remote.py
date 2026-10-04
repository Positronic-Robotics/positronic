"""Tests for remote dataset server and client."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import numpy as np
import pos3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import uvicorn
from fastapi.testclient import TestClient

from positronic.dataset.edits import EditedEpisode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.remote import RemoteDataset
from positronic.dataset.remote_server import server as remote_server
from positronic.dataset.signal import RECORDED_TIME, TIMELINE_METADATA_KEY, TIMELINES_KEY, SupportsEncodedRepresentation
from positronic.dataset.time import Time, TimeBounds
from positronic.dataset.utilities.migrate_remote import migrate_dataset, migrate_remote_dataset
from positronic.dataset.video import VIDEO_ENCODING_V1, VIDEO_ENCODING_V2, VideoSignal, VideoSignalWriter
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
                    ew.append(
                        'action',
                        np.array([i * 0.1, i * 0.2], dtype=np.float32),
                        Time(**{RECORDED_TIME: 1000 + i * 100}),
                    )

                # Video signal
                video_path = ew.path / 'cam.mp4'
                frames_path = ew.path / 'cam.frames.parquet'
                with VideoSignalWriter(video_path, frames_path, fps=30) as vw:
                    for i in range(3):
                        frame = np.full((64, 64, 3), (ep_idx + 1) * 50 + i * 10, dtype=np.uint8)
                        vw.append(frame, Time(**{RECORDED_TIME: 1000 + i * 100}))

    return LocalDataset(root)


@pytest.fixture
def test_client(dataset_with_video):
    """Create a FastAPI TestClient with the dataset."""
    remote_server._dataset = dataset_with_video
    return TestClient(remote_server._app)


# --- Server endpoint tests ---


def test_dataset_info_endpoint(test_client, dataset_with_video):
    r = test_client.get('/api/v2/dataset/info')
    assert r.status_code == 200
    data = r.json()
    assert data['num_episodes'] == 2
    assert 'meta' in data


def test_episode_info_endpoint(test_client):
    r = test_client.get('/api/v2/episodes/0/info')
    assert r.status_code == 200
    data = r.json()
    static = deserialize(bytes.fromhex(data['static']))
    assert static['task'] == 'task_0'
    assert static['episode_id'] == 0
    assert 'action' in data['signals']
    assert 'cam' in data['signals']
    assert data['signals']['action']['length'] == 5
    assert data['signals']['cam']['length'] == 3
    assert data['signals']['cam']['encoding_format'] == VIDEO_ENCODING_V2
    assert data['signals']['action'][TIMELINES_KEY] == [RECORDED_TIME]
    assert data['signals']['cam'][TIMELINES_KEY] == [RECORDED_TIME]


def test_episode_info_not_found(test_client):
    r = test_client.get('/api/v2/episodes/999/info')
    assert r.status_code == 404


def test_signal_timestamps_endpoint(test_client):
    # Test with indices
    r = test_client.post(
        '/api/v2/episodes/0/signals/action/timestamps', json={'indices': [0, 2, 4], TIMELINES_KEY: [RECORDED_TIME]}
    )
    assert r.status_code == 200
    data = r.json()
    assert data['timestamps'] == [[1000], [1200], [1400]]

    # Test with slice
    r = test_client.post(
        '/api/v2/episodes/0/signals/action/timestamps', json={'slice': [0, 3, None], TIMELINES_KEY: [RECORDED_TIME]}
    )
    assert r.status_code == 200
    data = r.json()
    assert data['timestamps'] == [[1000], [1100], [1200]]


@pytest.mark.parametrize(
    'timelines,status',
    [([], 400), ([''], 400), (['   '], 400), ([RECORDED_TIME, RECORDED_TIME], 400), (['missing'], 404)],
)
def test_signal_timestamps_reject_invalid_selectors(test_client, timelines, status):
    response = test_client.post(
        '/api/v2/episodes/0/signals/action/timestamps', json={TIMELINES_KEY: timelines, 'indices': [0]}
    )
    assert response.status_code == status
    assert response.json()['detail']


def test_signal_values_endpoint(test_client):
    r = test_client.post('/api/v2/episodes/0/signals/action/values', json={'indices': [0, 1]})
    assert r.status_code == 200
    values = deserialize(r.content)
    assert len(values) == 2
    np.testing.assert_allclose(values[0], [0.0, 0.0])
    np.testing.assert_allclose(values[1], [0.1, 0.2])


def test_signal_search_endpoint(test_client):
    r = test_client.post(
        '/api/v2/episodes/0/signals/action/search',
        json={TIMELINES_KEY: [RECORDED_TIME], 'timestamps': [[1050], [1150]]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data['indices'] == [0, 1]


def test_signal_encoded_endpoint(test_client):
    r = test_client.get('/api/v2/episodes/0/signals/cam/encoded')
    assert r.status_code == 200
    assert r.headers['x-encoding-format'] == VIDEO_ENCODING_V2
    assert len(r.content) > 0


def test_signal_encoded_not_supported(test_client):
    r = test_client.get('/api/v2/episodes/0/signals/action/encoded')
    assert r.status_code == 400


def test_episode_sample_endpoint(test_client):
    r = test_client.post(
        '/api/v2/episodes/0/sample', json={TIMELINES_KEY: [RECORDED_TIME], 'timestamps': [[1000], [1100]]}
    )
    assert r.status_code == 200
    data = r.json()
    assert deserialize(bytes.fromhex(data['static']))['task'] == 'task_0'
    assert 'action' in data['signals']
    action_values = deserialize(bytes.fromhex(data['signals']['action']['values']))
    assert len(action_values) == 2


# --- RemoteDataset client tests ---


def find_free_port():
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
            httpx.get(f'http://127.0.0.1:{port}/api/v2/dataset/info', timeout=0.1)
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


def test_edited_episode_over_remote_base(running_server):
    # The static overlay is backend-agnostic: it wraps any episode, including one served over HTTP.
    with RemoteDataset(running_server) as ds:
        edited = EditedEpisode(ds[0], {'verdict': 'success'})
        assert edited['verdict'] == 'success'
        assert edited['task'] == 'task_0'
        assert 'verdict' not in ds[1]


def test_remote_dataset_signal_access(running_server):
    with RemoteDataset(running_server) as ds:
        signal = ds[0]['action']
        assert len(signal) == 5
        value, ts = signal[0]
        np.testing.assert_allclose(value, [0.0, 0.0])
        assert ts == Time(**{RECORDED_TIME: 1000})


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
        result = ep.time[[Time(**{RECORDED_TIME: t}) for t in timestamps]]
        assert 'task' in result
        assert 'action' in result
        assert len(result['action']) == 2


def test_remote_dataset_video_encoded_stream(running_server):
    with RemoteDataset(running_server) as ds:
        signal = ds[0]['cam']
        assert signal.encoding_format == VIDEO_ENCODING_V2
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


@pytest.mark.parametrize('data', [42, np.zeros((32, 32, 3), dtype=np.uint8)], ids=['scalar', 'image'])
@pytest.mark.parametrize('remote', [False, True], ids=['local', 'remote'])
def test_migration_preserves_every_timeline(tmp_path, running_server, monkeypatch, data, remote):
    source_root = tmp_path / 'custom'
    dest_root = tmp_path / 'dest'
    with LocalDatasetWriter(source_root) as writer:
        with writer.new_episode() as episode:
            episode.append('signal', data, Time(world=1000, wall=2000))
    source = LocalDataset(source_root)
    monkeypatch.setattr(remote_server, '_dataset', source)
    with pos3.mirror(), RemoteDataset(running_server) as remote_source:
        dataset = remote_source if remote else source
        assert next(iter(dataset))['signal'].timelines == ('world', 'wall')
        migrate_dataset(dataset, str(dest_root))
    signal = LocalDataset(dest_root)[0]['signal']
    assert signal.timelines == ('world', 'wall')
    assert signal[0][1] == Time(world=1000, wall=2000)


def test_remote_timeline_metadata_is_required(running_server, monkeypatch):
    with RemoteDataset(running_server) as dataset:
        info = dataset._client.get_episode_info(0)
        del info['signals']['action'][TIMELINES_KEY]
        monkeypatch.setattr(dataset._client, 'get_episode_info', lambda index: info)
        with pytest.raises(KeyError):
            dataset[0]['action']


def test_migrate_remote_dataset_numeric_only(tmp_path):
    """Test migration of dataset with numeric signals only."""
    source_root = tmp_path / 'source'
    dest_root = tmp_path / 'dest'

    with LocalDatasetWriter(source_root) as w:
        for i in range(2):
            with w.new_episode() as ew:
                ew.set_static('id', i)
                for j in range(3):
                    ew.append('signal', np.array([j], dtype=np.float32), Time(**{RECORDED_TIME: 1000 + j * 100}))

    source_ds = LocalDataset(source_root)
    port = find_free_port()
    remote_server._dataset = source_ds

    config = uvicorn.Config(remote_server._app, host='127.0.0.1', port=port, log_level='error')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            httpx.get(f'http://127.0.0.1:{port}/api/v2/dataset/info', timeout=0.1)
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
    assert dest_ds[0].meta['uid'] == source_ds[0].meta['uid']
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
    assert ts == Time(**{RECORDED_TIME: 1000})


@pytest.mark.parametrize('remote', [False, True], ids=['local', 'remote'])
def test_legacy_video_encoding_and_migration(dataset_with_video, running_server, tmp_path, remote):
    camera = dataset_with_video[0]['cam']
    assert isinstance(camera, VideoSignal)
    legacy_index = pa.table({'ts_ns': [1000, 1100, 1200], 'ts_ns.capture': [30, 20, 10]}).replace_schema_metadata({
        TIMELINE_METADATA_KEY: b'world'
    })
    pq.write_table(legacy_index, camera.frames_index_path)
    destination = tmp_path / 'legacy_migrated'

    with pos3.mirror(), RemoteDataset(running_server) as remote_source:
        source = remote_source if remote else dataset_with_video
        assert source[0]['cam'].encoding_format == VIDEO_ENCODING_V1
        migrate_dataset(source, str(destination))

    copied = LocalDataset(destination)[0]['cam']
    assert isinstance(copied, VideoSignal)
    assert copied.encoding_format == VIDEO_ENCODING_V1
    assert copied.timelines == ('world',)
    assert list(copied.timestamps('world')) == [1000, 1100, 1200]
    assert copied.video_path.read_bytes() == camera.video_path.read_bytes()
    assert pq.read_table(copied.frames_index_path).equals(legacy_index, check_metadata=True)


def test_video_signal_supports_encoded_protocol(dataset_with_video):
    """Verify VideoSignal implements SupportsEncodedRepresentation."""
    ep = dataset_with_video[0]
    cam = ep['cam']
    assert isinstance(cam, SupportsEncodedRepresentation)
    assert cam.encoding_format == VIDEO_ENCODING_V2
    chunks = list(cam.iter_encoded_chunks())
    assert len(chunks) > 0


@pytest.fixture
def named_remote(tmp_path, running_server, monkeypatch):
    path = tmp_path / 'named'
    with LocalDatasetWriter(path) as writer:
        with writer.new_episode() as episode:
            episode.set_static('task', 'test')
            for value, ts in enumerate([Time(A=100, B=900), Time(A=140, B=950), Time(A=220, B=950)]):
                episode.append('left', value, ts)
            episode.append('right', 4, Time(A=100, C=1))
            episode.append('right', 5, Time(A=170, C=2))
            episode.append('separate', 6, Time(D=1))
    local = LocalDataset(path)
    monkeypatch.setattr(remote_server, '_dataset', local)
    with RemoteDataset(running_server) as remote:
        yield local, remote


def test_remote_named_query_parity(named_remote):
    local, remote = named_remote
    local_ep, remote_ep = local[0], remote[0]
    for query in [Time(A=150), Time(B=950), Time(B=950, A=150), Time(A=2**100), Time(A=150, B=950)]:
        assert remote_ep['left'].time[query] == local_ep['left'].time[query]
    assert remote_ep['left'].bounds(('B', 'A')) == local_ep['left'].bounds(('B', 'A'))
    assert remote_ep['left'].bounds('A') == local_ep['left'].bounds('A') == TimeBounds(100, 220)
    assert remote_ep['left'].bounds('A').start == 100
    assert remote_ep['left'].bounds('A').finish == 220
    assert remote_ep['left'].bounds(('A',)).start == Time(A=100)
    assert remote_ep['left'].bounds(('A',)).finish == Time(A=220)
    assert list(remote_ep['left'].timestamps('A')) == list(local_ep['left'].timestamps('A')) == [100, 140, 220]
    assert list(remote_ep['left'].timestamps(('A',))) == [Time(A=100), Time(A=140), Time(A=220)]
    assert remote_ep.bounds('A') == local_ep.bounds('A') == TimeBounds(100, 220)
    grid = [Time(A=100), Time(A=150), Time(A=200), Time(A=250)]
    assert list(remote_ep['left'].time[grid]) == list(local_ep['left'].time[grid])
    for query in [Time(A=150), Time(A=150, B=950), Time(D=1), Time(missing=1)]:
        assert remote_ep.time[query] == local_ep.time[query]
    for queries in [grid, [Time(A=150, B=950)], [Time(missing=1)], []]:
        actual, expected = remote_ep.time[queries], local_ep.time[queries]
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_equal(actual[name], expected[name])
    for episode in [remote_ep, local_ep]:
        with pytest.raises(KeyError):
            episode.time[[Time(A=99)]]
        with pytest.raises(KeyError):
            episode['left'].time[Time(A=-(2**100))]
        with pytest.raises(ValueError):
            episode.time[[Time(A=200), Time(A=100)]]


@pytest.mark.parametrize('indices', [[], slice(0, 0), slice(3, 3)])
@pytest.mark.parametrize('timelines', [('A',), ('B', 'A')])
def test_empty_remote_timestamps_preserve_selected_names(named_remote, indices, timelines):
    _, remote = named_remote
    times = remote._client.get_signal_timestamps(0, 'left', indices, timelines)
    assert times.timelines == timelines
    assert list(times) == []


def test_remote_discovery_does_not_load_values(named_remote, monkeypatch):
    local, remote = named_remote
    signal_type = type(local[0]['left'])

    def unexpected_read(*args):
        pytest.fail('Timeline discovery and bounds must not read values')

    with monkeypatch.context() as patch:
        patch.setattr(signal_type, '_values_at', unexpected_read)
        episode = remote[0]
        assert set(episode.signals) == {'left', 'right', 'separate'}
        assert episode.bounds(('A',)) == (Time(A=100), Time(A=220))
        assert episode.bounds('A') == TimeBounds(100, 220)
    assert episode['left'].dtype == np.dtype('int64')
    assert episode['left'].shape == ()


def test_named_remote_migration_preserves_coordinates(named_remote, tmp_path):
    local, remote = named_remote
    destination = tmp_path / 'migrated_named'
    with pos3.mirror():
        migrate_dataset(remote, str(destination))
    result = LocalDataset(destination)[0]
    for name, signal in local[0].signals.items():
        assert result[name].timelines == signal.timelines
        assert list(result[name]) == list(signal)
