"""Flux3 client recipes, session semantics and thin-server transport."""

import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from positronic_model_server import serialization, server_wire, websocket_wire
from positronic_model_server.protocol import AUTH_HEADER, bearer
from positronic_model_server.server import ModelServer
from positronic_wire import websocket, wire

from positronic import keys as obs_keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.drivers.roboarm import command
from positronic.offboard.client import InferenceClient
from positronic.policy import keys as policy_keys
from positronic.policy import spec
from positronic.policy.executor import Executor, WaitStatus
from positronic.vendors.flux3_action.serving import keys, recipe, server


class NativePolicy:
    def __init__(self, **config):
        self.config = SimpleNamespace(**{
            'fps': 15.0,
            'chunk_size': 32,
            'rtc_max_prefix': 15,
            'quantization': None,
            'action_parameterization': 'absolute',
            **config,
        })
        self.calls = []

    def predict(self, observation, committed_actions, *, seed):
        self.calls.append((observation, committed_actions, seed))
        actions = np.tile(np.arange(self.config.chunk_size, dtype=np.float32)[:, None], (1, 8))
        actions[:, -1] = 0.25
        if committed_actions is not None:
            actions[: len(committed_actions)] = committed_actions
        return actions

    def close(self):
        pass


@pytest.fixture
def rtc():
    return {keys.RTC.CALL_AFTER_SEC: 0.5, keys.RTC.MAX_DELAY_SEC: 1.0}


def model(native=None, rtc=None, *, three_cameras=False):
    return server.Flux3Model(
        NativePolicy() if native is None else native,
        metadata={},
        rtc=rtc,
        image_mappings=recipe.droid_cameras(three_cameras=three_cameras),
    )


def observation():
    return {
        obs_keys.JOINTS: np.arange(7, dtype=np.float32),
        obs_keys.GRIP: 0.25,
        obs_keys.TASK: 'Pick up the cup.',
        obs_keys.WRIST_IMAGE: np.full((360, 640, 3), 20, np.uint8),
        obs_keys.EXTERIOR_IMAGE: np.full((360, 640, 3), 60, np.uint8),
        obs_keys.EXTERIOR_IMAGE_2: np.full((360, 640, 3), 100, np.uint8),
    }


def request(targets=(), *, chunk_offset=0.0, observation_offset=0.0, elapsed=()):
    return {
        keys.RTC.FIELD: {
            keys.RTC.ACTION_PREFIX: {
                keys.RTC.PREFIX_ACTIONS: [{keys.ACTION: np.full(8, value, np.float32)} for value in targets],
                keys.RTC.CHUNK_OFFSET_SEC: chunk_offset,
                keys.RTC.OBSERVATION_OFFSET_SEC: observation_offset,
            },
            keys.CALL_TIMINGS: [{keys.ELAPSED_SEC: value} for value in elapsed],
        }
    }


def test_standalone_names_match_client_contract():
    for name in ('JOINTS', 'GRIP', 'TASK', 'WRIST_IMAGE', 'EXTERIOR_IMAGE', 'EXTERIOR_IMAGE_2'):
        assert getattr(keys, name) == getattr(obs_keys, name)
    for name in ('PLAN', 'CALL_TIMINGS', 'ELAPSED_SEC', 'ACTION_FPS'):
        assert getattr(keys, name) == getattr(policy_keys, name)
    for name in ('ACTION_PREFIX', 'PREFIX_ACTIONS', 'CHUNK_OFFSET_SEC', 'OBSERVATION_OFFSET_SEC'):
        assert getattr(keys.RTC, name) == getattr(policy_keys.RTC, name)


def test_droid_control_mode():
    assert recipe.DROID_CONTROL_MODE == command.to_wire(DROID_IMPEDANCE)


@pytest.mark.parametrize('three_cameras', [False, True])
@pytest.mark.parametrize('use_rtc', [False, True])
def test_recipe_preserves_cameras_state_commands_and_rtc_context(three_cameras, use_rtc, rtc):
    native = NativePolicy()
    served = model(native, rtc if use_rtc else None, three_cameras=three_cameras)
    session = served.prepare_session(served.parameters)
    spec.from_spec(session.client_stack)
    codec = spec.from_spec(recipe.inference(recipe.droid_cameras(three_cameras=three_cameras), rtc=use_rtc))
    obs = observation()
    if use_rtc:
        obs.update({keys.CALL_TIMINGS: [], **request()[keys.RTC.FIELD]})
    encoded = serialization.deserialise(serialization.serialise(codec.encode(obs)))
    result = codec.decode(session.infer(encoded))
    assert len(result) == 32
    np.testing.assert_array_equal(result[3][obs_keys.ROBOT_COMMAND].positions, np.full(7, 3))
    assert result[3][obs_keys.TARGET_GRIP] == 0.25
    np.testing.assert_array_equal(encoded[keys.JOINT_POSITION], np.arange(7))
    assert encoded[keys.GRIPPER_POSITION] == [0.25]
    assert encoded[keys.PROMPT] == obs[obs_keys.TASK]
    for name, value in ((keys.WRIST, 20), (keys.EXTERIOR, 60), (keys.EXTERIOR_2, 100 if three_cameras else 60)):
        assert encoded[name].shape == (360, 640, 3)
        assert np.all(encoded[name] == value)
    if use_rtc:
        obs[keys.RTC.ACTION_PREFIX] = {
            keys.RTC.PREFIX_ACTIONS: result[3:],
            keys.RTC.CHUNK_OFFSET_SEC: 3 / 15,
            keys.RTC.OBSERVATION_OFFSET_SEC: -1 / 15,
        }
        obs[keys.CALL_TIMINGS] = [{keys.ELAPSED_SEC: 0.2}]
        decoded = codec.decode(session.infer(codec.encode(obs)))
        np.testing.assert_array_equal(native.calls[-1][1][:, 0], [4, 5, 6])
        np.testing.assert_array_equal(decoded[0][obs_keys.ROBOT_COMMAND].positions, np.full(7, 4))


def test_rtc_uses_recent_maximum_latency_and_keeps_sessions_independent(rtc):
    native = NativePolicy()
    served = model(native, rtc)
    first = served.prepare_session(served.parameters)
    first.infer(request())
    first.infer(request(range(10), elapsed=[0.1, 0.2]))
    assert native.calls[-1][1].shape == (3, 8)
    first.infer(request(range(10), elapsed=[0.1] * 5))
    assert native.calls[-1][1].shape == (2, 8)
    second = served.prepare_session(served.parameters)
    second.infer(request())
    assert native.calls[-1][1] is None
    assert native.calls[-1][2] == 1


@pytest.mark.parametrize('offset', [0, -2.0])
def test_empty_or_exhausted_plan_is_unconditioned(rtc, offset):
    native = NativePolicy()
    served = model(native, rtc)
    session = served.prepare_session(served.parameters)
    session.infer(request())
    session.infer(request(() if offset == 0 else range(4), observation_offset=offset, elapsed=[0.1]))
    assert native.calls[-1][1] is None


@pytest.mark.parametrize('elapsed', [-1, float('nan'), float('inf'), True, 1.1])
def test_invalid_or_excessive_latency_fails_before_prediction(rtc, elapsed):
    native = NativePolicy()
    served = model(native, rtc)
    session = served.prepare_session(served.parameters)
    session.infer(request())
    with pytest.raises(ValueError):
        session.infer(request(range(5), elapsed=[elapsed]))
    assert len(native.calls) == 1


@pytest.mark.parametrize(
    'kwargs',
    [
        {'chunk_offset': -1},
        {'chunk_offset': 0.01},
        {'chunk_offset': 2.0},
        {'observation_offset': 0.1},
        {'observation_offset': float('nan')},
    ],
)
def test_bad_prefix_alignment_fails_before_prediction(rtc, kwargs):
    native = NativePolicy()
    served = model(native, rtc)
    session = served.prepare_session(served.parameters)
    session.infer(request())
    with pytest.raises(ValueError):
        session.infer(request(range(5), **kwargs))
    assert len(native.calls) == 1


@pytest.mark.parametrize(
    'config',
    [
        {'rtc_max_prefix': 0},
        {'quantization': 'fp8r'},
        {'action_parameterization': 'joint_delta'},
        {'rtc_max_prefix': 2},
    ],
)
def test_incompatible_checkpoint_cannot_start_rtc(config, rtc):
    with pytest.raises(ValueError):
        model(NativePolicy(**config), rtc)


def test_ordinary_model_cannot_enable_unwarmed_rtc(rtc):
    served = model(NativePolicy(rtc_max_prefix=0))
    session = served.prepare_session(served.parameters)
    with pytest.raises(ValueError, match='does not enable RTC'):
        session.infer(request())
    with pytest.raises(ValueError, match='Launch droid_rtc'):
        served.prepare_session({**served.parameters, keys.RTC.FIELD: rtc})


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('horizon_sec', -1),
        ('horizon_sec', True),
        ('jpeg_quality', 101),
        ('compress_images', 1),
        ('image_mappings', {}),
        (keys.RTC.FIELD, {}),
    ],
)
def test_invalid_session_settings_fail(name, value, rtc):
    served = model(rtc=rtc)
    with pytest.raises(ValueError):
        served.prepare_session({**served.parameters, name: value})


def test_rtc_refuses_playback_horizon_and_prediction_after_chunk_end(rtc):
    served = model(rtc=rtc)
    with pytest.raises(ValueError, match='horizon_sec=None'):
        served.prepare_session({**served.parameters, 'horizon_sec': 0.5})
    with pytest.raises(ValueError, match='leave actions'):
        served.prepare_session({**served.parameters, keys.RTC.FIELD: {**rtc, keys.RTC.CALL_AFTER_SEC: 3}})


@pytest.mark.parametrize('use_rtc', [False, True])
def test_load_warms_selected_inference_paths_and_preserves_checkpoint_metadata(monkeypatch, rtc, use_rtc):
    native = NativePolicy()
    constructor_args = []

    def load(*args, **kwargs):
        constructor_args.append((args, kwargs))
        return native

    monkeypatch.setitem(sys.modules, server.__package__ + '.native', SimpleNamespace(Policy=load))
    served = server.load_model(
        'checkpoint', 'revision', 'variant', image_mappings=recipe.droid_cameras(), rtc=rtc if use_rtc else None
    )
    assert constructor_args[0][0] == ('checkpoint', 'revision', 'variant')
    assert constructor_args[0][1]['rtc'] is use_rtc
    assert served.metadata['checkpoint_id'] == 'checkpoint'
    assert served.metadata['revision'] == 'revision'
    assert served.metadata['subfolder'] == 'variant'
    assert native.calls[0][1] is None
    if use_rtc:
        assert native.calls[-1][1].shape == (15, 8)
    session = served.prepare_session(served.parameters)
    session.infer(request() if use_rtc else {})
    assert native.calls[-1][1] is None
    assert native.calls[-1][2] == 1


def test_failed_warmup_releases_native_policy(monkeypatch):
    class FailingPolicy(NativePolicy):
        closed = False

        def predict(self, *args, **kwargs):
            raise RuntimeError('warmup failed')

        def close(self):
            self.closed = True

    native = FailingPolicy()
    monkeypatch.setitem(sys.modules, server.__package__ + '.native', SimpleNamespace(Policy=lambda *a, **kw: native))
    with pytest.raises(RuntimeError, match='warmup failed'):
        server.load_model('checkpoint', None, None, image_mappings=recipe.droid_cameras(), rtc=None)
    assert native.closed


def test_thin_server_round_trip():
    native = NativePolicy()
    served = ModelServer(lambda: model(native), auth_token='test-token')
    transport = websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))
    ready = threading.Event()
    thread = threading.Thread(target=served.serve, args=([transport], ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(10)
        address = wire.HostPortAddress('127.0.0.1', transport.served_address.port, wire.SESSION_PATH, '')
        client = InferenceClient(websocket.WebsocketClientWire(), address, headers={AUTH_HEADER: bearer('test-token')})
        with closing(client.new_session()) as session:
            codec = spec.from_spec(recipe.inference(recipe.droid_cameras()))
            actions = codec.decode(session.infer(codec.encode(observation())))
            assert len(actions) == 32
            assert actions[4][obs_keys.TARGET_GRIP] == 0.25
    finally:
        served.shutdown()
        thread.join(10)
        assert not thread.is_alive()


def test_rtc_schedule_keeps_playing_during_next_prediction(rtc):
    native = NativePolicy()
    served = model(native, rtc)
    session = served.prepare_session({**served.parameters, 'compress_images': False})
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    pending = threading.Event()
    release = threading.Event()

    def infer(obs):
        if native.calls:
            pending.set()
            assert release.wait(5)
        return session.infer(obs)

    run = runtime.start(spec.from_spec(session.client_stack), infer)
    obs = observation()
    try:
        run.send(obs)
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 100_000_000
        assert run.send(obs).commands[obs_keys.ROBOT_COMMAND].positions[0] == 0
        now[0] = 600_000_000
        run.send(obs)
        assert pending.wait(2)
        now[0] = 700_000_000
        assert run.send(obs).commands[obs_keys.ROBOT_COMMAND].positions[0] > 0
        release.set()
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 800_000_000
        run.send(obs)
        assert len(native.calls) == 2
        assert native.calls[-1][0][keys.RTC.FIELD][keys.RTC.ACTION_PREFIX][keys.RTC.PREFIX_ACTIONS]
    finally:
        release.set()
        run.close()
        runtime.close()


def test_standalone_cli_imports_without_positronic_or_flux(tmp_path):
    script = """
import importlib.abc
import sys
class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'positronic', 'pimm', 'flux_action', 'torch'}:
            raise AssertionError(f'Standalone adapter imports {fullname}')
sys.meta_path.insert(0, ForbiddenImports())
sys.path.insert(0, sys.argv[1])
from serving import server
sys.argv = ['flux3_serving.server', 'droid_rtc', '--help']
server.main()
"""
    subprocess.run([sys.executable, '-I', '-c', script, str(Path(server.__file__).parents[1])], check=True)
