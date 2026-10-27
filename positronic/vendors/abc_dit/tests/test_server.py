"""The ABC-DiT server, run against a stand-in for ABC's interpreter and its policy server."""

import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from positronic_model_server import server_wire, websocket_wire
from positronic_model_server.protocol import AUTH_HEADER, bearer
from positronic_wire import websocket, wire
from positronic_wire.roboarena import TextAnswer

from positronic import keys
from positronic.offboard import roboarena
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.vendors import abc_dit
from positronic.vendors.abc_dit import codecs, server
from positronic.vendors.abc_dit.tests import abc_stand_in

TOKEN = 'run-token'
CHECKPOINT = '/checkpoints/abc_dit_xl_200k_model.pt'
LOG = 'backend.jsonl'


def _abc_python(directory: Path, mode: abc_stand_in.Mode) -> str:
    """An interpreter whose run of ABC's policy server is the stand-in, answering in `mode`."""
    wrapper = directory / 'python'
    stand_in = abc_stand_in.__file__
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stand_in}" "{directory / LOG}" {mode.value} "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)


def _log(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / LOG).read_text().splitlines()]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _session(
    policy_server: PolicyServer, obs: dict[str, Any], query: str = '', calls: int = 1
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """The metadata and the commands of `calls` inferences of one session on `query`, behind the run's token."""
    wires = [websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))]
    ready = threading.Event()
    thread = threading.Thread(target=policy_server.serve, args=(wires, ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=30)
        served = wires[0].served_address
        assert isinstance(served, server_wire.ServedHostPort)
        address = wire.HostPortAddress('127.0.0.1', served.port, wire.SESSION_PATH, query)
        client = InferenceClient(websocket.WebsocketClientWire(), address, headers={AUTH_HEADER: bearer(TOKEN)})
        client.keepalive()
        session = client.new_session()
        try:
            return session.metadata, [session.infer(obs) for _ in range(calls)]
        finally:
            session.close()
    finally:
        policy_server.shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive()


def _infer_in_a_session(policy_server: PolicyServer, obs: dict[str, Any]) -> list[dict[str, Any]]:
    """The commands one session gets for `obs`, from `policy_server` behind the run's token."""
    return _session(policy_server, obs)[1][0]


def _view(level: int) -> np.ndarray:
    return np.full((260, 416, 3), level, dtype=np.uint8)


def test_a_session_reaches_abcs_server_in_its_fields_and_gets_the_whole_chunk(tmp_path):
    port = _free_port()
    model = server.abc_dit_model.override(
        checkpoint=CHECKPOINT, abc_python=_abc_python(tmp_path, abc_stand_in.Mode.SERVE), backend_port=port
    )
    obs = {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 1.0,
        keys.EXTERIOR_IMAGE: _view(40),
        keys.WRIST_LEFT_IMAGE: _view(120),
        keys.WRIST_RIGHT_IMAGE: _view(200),
        keys.TASK: 'put the plastic bottles in the bin',
    }

    actions = _infer_in_a_session(PolicyServer(model, server.pipeline, auth_token=TOKEN), obs)

    assert len(actions) == len(abc_stand_in.CHUNK)
    for action, expected in zip(actions, abc_stand_in.CHUNK, strict=True):
        np.testing.assert_allclose(action[keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)].positions, expected[:6])
        np.testing.assert_allclose(
            action[keys.arm_channel(keys.ROBOT_COMMAND, keys.RIGHT_ARM)].positions, expected[7:13]
        )
        assert action[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == pytest.approx(1.0 - expected[6])
        assert action[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == pytest.approx(1.0 - expected[13])
    # One warm-up request comes before the session's own.
    started, _, request = _log(tmp_path)
    fields = request['request']
    images = fields.pop('images')
    # The session's request carries the first noise draw of ABC's sim-eval seed.
    first_draw = np.random.default_rng(0).standard_normal((30, 14), dtype=np.float32)
    assert fields['noise'].pop('sum') == pytest.approx(float(first_draw.sum()), rel=1e-5)
    # rules-allow: hardcoded-keys — the fields of ABC's policy server as its code spells them, pinned apart from the
    # package constants so a typo in a constant fails here.
    assert fields == {
        'state': [*range(6), 1.0, *range(6, 12), 0.0],
        'prompt': 'sim throw plastic bottles in bin',
        'noise': {'shape': [30, 14], 'dtype': 'float32'},
        roboarena.ENDPOINT: roboarena.INFER,
    }
    assert {name: view['shape'] for name, view in images.items()} == dict.fromkeys(
        ('top', 'left', 'right'), [3, 168, 224]
    )
    # The 416x260 views pad to 224x168 with black rows, so each mean is 140/168 of the view's level.
    means = {name: round(view['mean'] * 168 / 140) for name, view in images.items()}
    assert means == pytest.approx({'top': 40, 'left': 120, 'right': 200}, abs=2)
    assert started['argv'][:2] == ['-P', str(server.BACKEND_SCRIPT)]
    assert set(started['argv'][2:]) == {
        f'--policy.checkpoint-path={CHECKPOINT}',
        '--policy-type=dit',
        '--policy.fast-inference',
        f'--port={port}',
    }
    assert not _alive(started['pid']), 'the server left the backend running after it stopped'


def test_the_warm_up_leaves_the_seeded_noise_stream_to_the_sessions(tmp_path):
    model = server.abc_dit_model(
        checkpoint=CHECKPOINT,
        abc_python=_abc_python(tmp_path, abc_stand_in.Mode.SERVE),
        noise_seed=7,
        backend_port=_free_port(),
    )
    try:
        model(server._warm_observation(), session_id='first')
        model(server._warm_observation(), session_id='second')
    finally:
        model.close()
    _, _, first, second = _log(tmp_path)
    draws = np.random.default_rng(7)
    for logged in (first, second):
        expected = draws.standard_normal((30, 14), dtype=np.float32)
        assert logged['request']['noise']['sum'] == pytest.approx(float(expected.sum()), rel=1e-5)


def _time_wait_port() -> int:
    """A port whose only socket is the TIME_WAIT end of a connection its listener closed first."""
    with socket.create_server(('127.0.0.1', 0)) as listener:
        port = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1', port)) as client:
            accepted, _ = listener.accept()
            accepted.close()
            client.recv(1)
    return port


def test_a_backend_port_another_server_listens_on_fails_the_load_before_the_backend_starts(tmp_path):
    with socket.create_server(('127.0.0.1', 0)) as occupant:
        with pytest.raises(OSError, match='Address already in use'):
            server.abc_dit_model(
                checkpoint=CHECKPOINT,
                abc_python=_abc_python(tmp_path, abc_stand_in.Mode.SERVE),
                backend_port=occupant.getsockname()[1],
            )
    assert not (tmp_path / LOG).exists()


def test_a_backend_port_with_only_closed_connections_on_it_loads(tmp_path):
    model = server.abc_dit_model(
        checkpoint=CHECKPOINT, abc_python=_abc_python(tmp_path, abc_stand_in.Mode.SERVE), backend_port=_time_wait_port()
    )
    model.close()


def test_a_load_without_fast_inference_asks_the_backend_for_none(tmp_path):
    model = server.abc_dit_model(
        checkpoint=CHECKPOINT,
        abc_python=_abc_python(tmp_path, abc_stand_in.Mode.SERVE),
        fast_inference=False,
        backend_port=_free_port(),
    )
    model.close()
    argv = _log(tmp_path)[0]['argv']
    assert '--policy.no-fast-inference' in argv and '--policy.fast-inference' not in argv


def test_a_warm_up_the_backend_refuses_fails_the_load_and_stops_the_backend(tmp_path):
    with pytest.raises(TextAnswer, match='the policy refused the request'):
        server.abc_dit_model(
            checkpoint=CHECKPOINT, abc_python=_abc_python(tmp_path, abc_stand_in.Mode.REFUSE), backend_port=_free_port()
        )
    assert not _alive(_log(tmp_path)[0]['pid'])


def test_a_backend_that_exits_fails_the_load(tmp_path):
    exits = tmp_path / 'python'
    exits.write_text('#!/bin/sh\nexit 3\n')
    exits.chmod(0o755)
    with pytest.raises(RuntimeError, match='exited with code 3'):
        server.abc_dit_model(checkpoint=CHECKPOINT, abc_python=str(exits), backend_port=_free_port())


def test_the_yambox_crop_pipeline_sends_each_cropped_view_to_abc_without_pad(tmp_path):
    model = server.abc_dit_model.override(
        checkpoint=CHECKPOINT, abc_python=_abc_python(tmp_path, abc_stand_in.Mode.SERVE), backend_port=_free_port()
    )
    # A 960x600 frame whose crop box is white and whose rest is black: a view that misses the box shows black.
    views = {}
    for key, box in codecs.YAMBOX_CROPS.items():
        frame = np.zeros((600, 960, 3), dtype=np.uint8)
        frame[round(box.top * 600) : round(box.bottom * 600), round(box.left * 960) : round(box.right * 960)] = 255
        views[key] = frame
    obs = {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 1.0,
        **views,
        keys.TASK: 'turn mug right side up',
    }

    _infer_in_a_session(PolicyServer(model, server.yambox_crop_pipeline, auth_token=TOKEN), obs)

    request = _log(tmp_path)[-1]['request']
    assert request['prompt'] == 'sim turn mug right side up'
    for view in request['images'].values():
        assert view['shape'] == [3, 168, 224]
        assert view['mean'] > 250


def _yambox_obs() -> dict[str, Any]:
    frame = np.zeros((600, 960, 3), dtype=np.uint8)
    return {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 1.0,
        keys.EXTERIOR_IMAGE: frame,
        keys.WRIST_LEFT_IMAGE: frame,
        keys.WRIST_RIGHT_IMAGE: frame,
        keys.TASK: 'load plates into tabletop dish rack',
    }


def _prefix_model(directory: Path):
    return server.model_with_prefix.override(
        checkpoint=CHECKPOINT, abc_python=_abc_python(directory, abc_stand_in.Mode.SERVE), backend_port=_free_port()
    )


def test_a_session_query_sets_the_crop_the_schedule_and_the_prompt_prefix_and_the_metadata_names_them(tmp_path):
    query = 'crop=yambox_wide_top&schedule=rtc&sim_prefix=false'
    policy_server = PolicyServer(_prefix_model(tmp_path), server.yambox_crop_pipeline, auth_token=TOKEN)

    meta, replies = _session(policy_server, _yambox_obs(), query, calls=2)

    assert meta['effective_params'] == {'crop': 'yambox_wide_top', 'schedule': 'rtc', 'sim_prefix': False}
    schedule = meta['local_stack']['seq'][0]
    assert schedule['name'] == 'chunked_schedule'
    assert schedule['args'] == {
        'fps': 30.0,
        'horizon_sec': pytest.approx(16 / 30),
        'inference_lead': pytest.approx(7 / 30),
    }
    assert meta[codecs.CROPS_META][keys.EXTERIOR_IMAGE] == [0.0, 0.0, 1.0, 1.0]
    assert meta[codecs.PROMPT_PREFIX_META] == ''
    assert meta[codecs.PREFIX_LENGTH_META] == 4
    # Each reply holds the 16 actions after the 4 of the prefix.
    for reply in replies:
        assert len(reply) == 16
        np.testing.assert_allclose(
            reply[0][keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)].positions, abc_stand_in.CHUNK[4, :6]
        )
    first, second = (entry['request'] for entry in _log(tmp_path)[-2:])
    assert first['prompt'] == 'load plates into tabletop dish rack'
    assert abc_dit.EXECUTED_ACTIONS not in first
    # The first prefix is the state, four times; the second is the last four actions the client played.
    state = [*range(6), 1.0, *range(6, 12), 0.0]
    assert first[abc_dit.PREFIX_LENGTH] == 4 and first[abc_dit.ACTION_PREFIX]['shape'] == [4, 14]
    assert first[abc_dit.ACTION_PREFIX]['sum'] == pytest.approx(4 * sum(state))
    assert second[abc_dit.ACTION_PREFIX]['sum'] == pytest.approx(float(abc_stand_in.CHUNK[16:20].sum()), rel=1e-5)


def test_a_session_with_no_query_gets_the_launch_pipeline(tmp_path):
    policy_server = PolicyServer(_prefix_model(tmp_path), server.yambox_crop_pipeline, auth_token=TOKEN)

    meta, (reply,) = _session(policy_server, _yambox_obs())

    assert meta['effective_params'] == {'crop': 'yambox', 'schedule': 'sync', 'sim_prefix': True}
    assert meta['local_stack']['seq'][0]['args'] == {'fps': 30.0, 'horizon_sec': 0.5}
    assert len(reply) == len(abc_stand_in.CHUNK)
    request = _log(tmp_path)[-1]['request']
    assert request['prompt'] == 'sim load plates into tabletop dish rack'
    assert abc_dit.PREFIX_LENGTH not in request and abc_dit.ACTION_PREFIX not in request


@pytest.mark.parametrize('query', ['crop=wider', 'schedule=async', 'sim_prefix=maybe', 'fps=10'])
def test_a_session_query_with_an_unknown_value_or_name_is_refused(tmp_path, query):
    policy_server = PolicyServer(_prefix_model(tmp_path), server.yambox_crop_pipeline, auth_token=TOKEN)

    with pytest.raises(RuntimeError, match='Server error'):
        _session(policy_server, _yambox_obs(), query)


def test_a_backend_with_a_prefix_warms_it_and_refuses_a_prefix_of_another_length(tmp_path):
    model = _prefix_model(tmp_path).instantiate()
    request = {**server._warm_observation(), abc_dit.PREFIX_LENGTH: 5, abc_dit.EXECUTED_ACTIONS: 16}
    try:
        with pytest.raises(ValueError, match='serves an action prefix of 4, not 5'):
            model(request, session_id='session')
    finally:
        model.close()
    assert f'--policy.rtc-prefix-length={server.PREFIX_LENGTH}' in _log(tmp_path)[0]['argv']


def test_an_ended_session_starts_its_next_prefix_from_the_state(tmp_path):
    model = _prefix_model(tmp_path).instantiate()
    request = {**server._warm_observation(), abc_dit.PREFIX_LENGTH: 4, abc_dit.EXECUTED_ACTIONS: 16}
    try:
        model(request, session_id='a')
        model(request, session_id='b')
        model.end_session('a')
        model(request, session_id='a')
        model(request, session_id='b')
    finally:
        model.close()
    sums = [entry['request'][abc_dit.ACTION_PREFIX]['sum'] for entry in _log(tmp_path)[-4:]]
    played = pytest.approx(float(abc_stand_in.CHUNK[16:20].sum()), rel=1e-5)
    # The warm state is zero, so a prefix from the state sums to zero.
    assert sums == [0.0, 0.0, 0.0, played]
