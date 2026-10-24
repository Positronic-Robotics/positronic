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
from positronic.vendors.abc_dit import server
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


def _infer_in_a_session(policy_server: PolicyServer, obs: dict[str, Any]) -> list[dict[str, Any]]:
    """The commands one session gets for `obs`, from `policy_server` behind the run's token."""
    wires = [websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))]
    ready = threading.Event()
    thread = threading.Thread(target=policy_server.serve, args=(wires, ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=30)
        served = wires[0].served_address
        assert isinstance(served, server_wire.ServedHostPort)
        address = wire.HostPortAddress('127.0.0.1', served.port, wire.SESSION_PATH, '')
        client = InferenceClient(websocket.WebsocketClientWire(), address, headers={AUTH_HEADER: bearer(TOKEN)})
        client.keepalive()
        session = client.new_session()
        try:
            return session.infer(obs)
        finally:
            session.close()
    finally:
        policy_server.shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive()


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
