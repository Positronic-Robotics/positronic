"""The FLUX 3 Action server, run against a stand-in for BFL's interpreter and its RoboLab server."""

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
from positronic.vendors.flux3_action import server
from positronic.vendors.flux3_action.tests import bfl_stand_in

TOKEN = 'run-token'
REPOSITORY = 'black-forest-labs/flux-3-action-droid'
REVISION = 'policy-commit'
SUBFOLDER = 'variants/gd'
LOG = 'backend.jsonl'


def _flux_python(directory: Path, mode: bfl_stand_in.Mode) -> str:
    """An interpreter whose run of `backend.py` is the stand-in, answering in `mode`."""
    wrapper = directory / 'python'
    stand_in = bfl_stand_in.__file__
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


def test_a_session_reaches_bfls_server_in_its_fields_and_gets_the_whole_chunk(tmp_path):
    port = _free_port()
    model = server.flux3_action_model.override(
        checkpoint=REPOSITORY,
        revision=REVISION,
        subfolder=SUBFOLDER,
        flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE),
        backend_port=port,
    )
    # The default pipeline serves an eval that renders one exterior view, as MolmoSpaces does.
    obs = {
        keys.WRIST_IMAGE: np.zeros((720, 1280, 3), dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.zeros((480, 640, 3), dtype=np.uint8),
        keys.JOINTS: np.arange(7, dtype=np.float32),
        keys.GRIP: 0.25,
        keys.TASK: 'put the cube in the bowl',
    }

    actions = _infer_in_a_session(PolicyServer(model, server.pipeline, auth_token=TOKEN), obs)

    assert len(actions) == len(bfl_stand_in.CHUNK)
    for action, expected in zip(actions, bfl_stand_in.CHUNK, strict=True):
        np.testing.assert_allclose(action[keys.ROBOT_COMMAND].positions, expected[:7])
        assert action[keys.TARGET_GRIP] == pytest.approx(expected[7])
    # One warm-up request comes before the session's own.
    started, _warm_up, request = _log(tmp_path)
    # rules-allow: hardcoded-keys — BFL's RoboLab fields as its documentation spells them, pinned apart from the
    # package constants so a typo in a constant fails here.
    assert request['request'] == {
        'observation/wrist_image_left': [360, 640, 3],
        'observation/exterior_image_1_left': [360, 640, 3],
        'observation/exterior_image_2_left': [360, 640, 3],
        'observation/joint_position': list(range(7)),
        'observation/gripper_position': [0.25],
        'prompt': 'put the cube in the bowl',
        roboarena.ENDPOINT: roboarena.INFER,
    }
    assert started['argv'][:2] == ['-P', str(server.BACKEND_SCRIPT)]
    assert dict(zip(started['argv'][2::2], started['argv'][3::2], strict=True)) == {
        '--checkpoint': REPOSITORY,
        '--port': str(port),
        '--revision': REVISION,
        '--subfolder': SUBFOLDER,
    }
    assert not _alive(started['pid']), 'the server left the backend running after it stopped'


def test_a_local_policy_directory_reaches_the_backend_with_no_revision_and_no_subfolder(tmp_path):
    model = server.flux3_action_model(
        checkpoint=str(tmp_path), flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.SERVE), backend_port=_free_port()
    )
    model.close()
    argv = _log(tmp_path)[0]['argv']
    assert dict(zip(argv[2::2], argv[3::2], strict=True)).keys() == {'--checkpoint', '--port'}


def test_a_warm_up_the_backend_refuses_fails_the_load_and_stops_the_backend(tmp_path):
    with pytest.raises(TextAnswer, match='the policy refused the request'):
        server.flux3_action_model(
            checkpoint=REPOSITORY,
            flux_python=_flux_python(tmp_path, bfl_stand_in.Mode.REFUSE),
            backend_port=_free_port(),
        )
    assert not _alive(_log(tmp_path)[0]['pid'])


def test_a_backend_that_exits_fails_the_load(tmp_path):
    exits = tmp_path / 'python'
    exits.write_text('#!/bin/sh\nexit 3\n')
    exits.chmod(0o755)
    with pytest.raises(RuntimeError, match='exited with code 3'):
        server.flux3_action_model(checkpoint=REPOSITORY, flux_python=str(exits), backend_port=_free_port())
