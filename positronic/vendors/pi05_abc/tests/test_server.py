"""The pi0.5 bottles server, run against a stand-in for the author's openpi server in the fork's interpreter."""

import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip('openpi_client')

from positronic_model_server import server_wire, websocket_wire
from positronic_model_server.protocol import AUTH_HEADER, bearer
from positronic_wire import websocket, wire

from positronic import keys
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.vendors.abc_dit import codecs as abc_codecs
from positronic.vendors.pi05_abc import server
from positronic.vendors.pi05_abc.tests import author_server_stand_in

TOKEN = 'run-token'
LOG = 'backend.jsonl'


def _fork(directory: Path) -> Path:
    """A fork checkout whose interpreter runs the stand-in in place of the author's server."""
    fork = directory / 'fork'
    (fork / server.FORK_OPENPI).mkdir(parents=True)
    python = fork / server.FORK_PYTHON
    python.parent.mkdir(parents=True)
    stand_in = author_server_stand_in.__file__
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stand_in}" "{directory / LOG}" "$@"\n')
    python.chmod(0o755)
    return fork


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


def _infer_in_a_session(policy_server: PolicyServer, obs: dict[str, Any], query: str = '') -> list[dict[str, Any]]:
    """The commands one session on `query` gets for `obs`, from `policy_server` behind the run's token."""
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
            return session.infer(obs)
        finally:
            session.close()
    finally:
        policy_server.shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive()


def _obs(task: str = server.TASK) -> dict[str, Any]:
    return {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 1.0,
        keys.EXTERIOR_IMAGE: np.full((300, 400, 3), 40, dtype=np.uint8),
        keys.WRIST_LEFT_IMAGE: np.full((300, 400, 3), 120, dtype=np.uint8),
        keys.WRIST_RIGHT_IMAGE: np.full((300, 400, 3), 200, dtype=np.uint8),
        keys.TASK: task,
    }


def test_a_session_reaches_the_authors_server_in_its_fields_and_gets_the_whole_chunk(tmp_path):
    checkpoint = tmp_path / 'pi05-abc-130k-task-finetune-16000'
    fork = _fork(tmp_path)
    port = _free_port()
    model = server.pi05_abc_model.override(checkpoint=str(checkpoint), fork=str(fork), port=port)

    actions = _infer_in_a_session(PolicyServer(model, server.pipeline, auth_token=TOKEN), _obs(), 'crop=none')

    assert len(actions) == len(author_server_stand_in.CHUNK)
    for action, expected in zip(actions, author_server_stand_in.CHUNK, strict=True):
        np.testing.assert_allclose(action[keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)].positions, expected[:6])
        np.testing.assert_allclose(
            action[keys.arm_channel(keys.ROBOT_COMMAND, keys.RIGHT_ARM)].positions, expected[7:13]
        )
        assert action[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == pytest.approx(1.0 - expected[6])
        assert action[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == pytest.approx(1.0 - expected[13])
    # One warm-up request comes before the session's own.
    started, _, request = _log(tmp_path)
    fields = request['request']
    # rules-allow: hardcoded-keys — the fields and flags of the author's server as its code spells them, pinned apart
    # from the package constants so a typo in a constant fails here.
    images = fields.pop('images')
    assert fields == {'state': [*range(6), 1.0, *range(6, 12), 0.0], 'prompt': 'put the plastic bottles in the bin'}
    # The 4:3 views fill ABC's 224x168 view, channels first, which openpi pads to its square input.
    assert {name: view['shape'] for name, view in images.items()} == dict.fromkeys(
        ('top', 'left', 'right'), [3, 168, 224]
    )
    assert {name: round(view['mean']) for name, view in images.items()} == {'top': 40, 'left': 120, 'right': 200}
    assert started['cwd'] == str(fork / 'third_party/policy/openpi')
    assert started['argv'] == [
        str(checkpoint / 'inference/openpi_server.py'),
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        '--config',
        'pi05_abc130k',
        '--checkpoint',
        str(checkpoint),
        '--norm-stats-path',
        str(checkpoint / 'assets/put_the_plastic_bottles_in_the_bin/norm_stats.json'),
        '--prompt',
        'put the plastic bottles in the bin',
        '--image-key-map',
        'top=image,left=left_wrist,right=right_wrist',
        '--flatten-prefix',
        'observation/',
        '--state-key',
        'observation/state',
    ]
    assert not _alive(started['pid']), 'the server left the backend running after it stopped'


def test_a_task_other_than_the_bottles_is_refused():
    codec = server.pipeline().codec
    with pytest.raises(ValueError, match='no prompt for the task'):
        codec.encode(_obs('sweep away the paper scraps from the table'))


@pytest.mark.parametrize('crop', list(abc_codecs.Crop))
def test_a_session_crop_names_the_boxes_of_that_set(crop):
    meta = server.pipeline(crop=crop.value).codec.meta
    crops = abc_codecs.CROPS[crop]
    assert meta[abc_codecs.CROPS_META] == {key: [b.left, b.top, b.right, b.bottom] for key, b in crops.items()}


def test_an_unknown_crop_is_refused():
    with pytest.raises(ValueError, match='Unknown crop'):
        server.pipeline(crop='center')


def test_the_default_crop_keeps_the_whole_exterior_frame():
    assert server.pipeline().codec.meta[abc_codecs.CROPS_META][keys.EXTERIOR_IMAGE] == [0.0, 0.0, 1.0, 1.0]
