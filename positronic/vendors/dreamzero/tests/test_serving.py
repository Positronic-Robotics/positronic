import gc
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from positronic_model_server import serialization

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.drivers.roboarm import command
from positronic.policy import spec
from positronic.policy.codecs import Codec
from positronic.policy.executor import Executor, WaitStatus
from positronic.vendors.dreamzero.serving import recipe, server
from positronic.vendors.dreamzero.serving.keys import (
    ACTION_GRIP,
    ACTION_JOINTS,
    CAMERAS,
    GRIP,
    JOINTS,
    OBS_TIME_NS,
    PROMPT,
)
from positronic.vendors.dreamzero.serving.server import DreamZeroModel


class NativePolicy:
    def __init__(self):
        self.observations = []
        self.resets = 0

    def predict(self, observation):
        self.observations.append(observation)
        return {
            ACTION_JOINTS: np.full((24, 7), len(self.observations) - 1, np.float32),
            ACTION_GRIP: np.ones((24, 1), np.float32),
        }

    def reset(self):
        self.resets += 1


def observation(index):
    return {
        **{name: np.full((2, 3, 3), index % 256, np.uint8) for name in CAMERAS},
        OBS_TIME_NS: round(index * 1e9 / 15),
        JOINTS: np.zeros((1, 7), np.float32),
        GRIP: np.zeros((1, 1), np.float32),
        PROMPT: 'Put banana in bowl.',
    }


def test_history_only_calls_and_selected_frames():
    native = NativePolicy()
    model = DreamZeroModel(native, width=3, height=2)
    session = model.prepare_session(model.parameters)
    assert session.infer(observation(0)) is not None
    assert native.observations[-1][CAMERAS[0]].shape == (1, 2, 3, 3)
    for index in range(1, 24):
        assert session.infer(observation(index)) is None
    assert session.infer(observation(24)) is not None
    np.testing.assert_array_equal(native.observations[-1][CAMERAS[0]][:, 0, 0, 0], [1, 8, 16, 24])
    session.close()


def test_late_samples_use_latest_observation_without_catching_up_predictions():
    native = NativePolicy()
    model = DreamZeroModel(native, width=3, height=2)
    session = model.prepare_session(model.parameters)
    session.infer(observation(0))
    assert session.infer(observation(100)) is not None
    assert session.infer(observation(101)) is None
    assert len(native.observations) == 2
    np.testing.assert_array_equal(native.observations[-1][CAMERAS[0]][:, 0, 0, 0], [0, 0, 0, 100])


@pytest.mark.parametrize('sample_fps', [15, 150])
def test_history_releases_old_images_and_close_releases_all_images(sample_fps):
    native = NativePolicy()
    model = DreamZeroModel(native, width=3, height=2)
    session = model.prepare_session({**model.parameters, 'prediction_period_sec': 10000})
    references = []
    for index in range(2000):
        frame = observation(index)
        frame[OBS_TIME_NS] = round(index * 1e9 / sample_fps)
        references.append(weakref.ref(frame[CAMERAS[0]]))
        session.infer(frame)
    del frame
    gc.collect()
    assert sum(reference() is not None for reference in references) <= 25
    assert not any(reference() is not None for reference in references[:-25])
    assert references[-1]() is not None
    session.close()
    gc.collect()
    assert not any(reference() is not None for reference in references)


def test_sessions_are_exclusive_and_restart_native_history():
    native = NativePolicy()
    model = DreamZeroModel(native, width=3, height=2)
    first = model.prepare_session(model.parameters)
    with pytest.raises(RuntimeError, match='one active session'):
        model.prepare_session(model.parameters)
    first.infer(observation(0))
    first.close()
    resets = native.resets
    first.close()
    assert native.resets == resets
    with pytest.raises(RuntimeError, match='closed'):
        first.infer(observation(1))
    second = model.prepare_session(model.parameters)
    assert second.infer(observation(0)) is not None
    assert native.observations[-1][CAMERAS[0]].shape[0] == 1
    second.close()


@pytest.mark.parametrize('mode', list(recipe.Mode))
def test_client_recipe_builds_and_native_arrays_decode_to_commands(mode):
    native = NativePolicy()
    model = DreamZeroModel(native, width=3, height=2)
    session = model.prepare_session({**model.parameters, 'mode': mode.value})
    spec.from_spec(session.client_stack)
    codec = spec.from_spec(recipe.inference(recipe.droid_cameras(), 3, 2, 90))
    assert isinstance(codec, Codec)
    raw = {
        keys.JOINTS: np.arange(7),
        keys.GRIP: 0.5,
        keys.TASK: 'Put banana in bowl.',
        OBS_TIME_NS: 0,
        **{source: np.zeros((2, 3, 3), np.uint8) for source in recipe.droid_cameras().values()},
    }
    encoded = serialization.deserialise(serialization.serialise(codec.encode(raw)))
    decoded = codec.decode(session.infer(encoded))
    assert len(decoded) == 24
    np.testing.assert_array_equal(decoded[0][keys.ROBOT_COMMAND].positions, np.zeros(7))
    assert decoded[0][keys.TARGET_GRIP] == 1.0
    np.testing.assert_array_equal(native.observations[-1][JOINTS], np.arange(7).reshape(1, 7))


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('sampling_period_sec', 0),
        ('prediction_period_sec', float('inf')),
        ('fps', True),
        ('history_offsets_sec', [-1, -2, -3, 0]),
        ('jpeg_quality', 101),
        ('mode', 'unknown'),
    ],
)
def test_invalid_session_configuration_does_not_claim_model(name, value):
    model = DreamZeroModel(NativePolicy(), width=3, height=2)
    with pytest.raises(ValueError):
        model.prepare_session({**model.parameters, name: value})
    model.prepare_session(model.parameters).close()


def test_observation_time_must_advance():
    model = DreamZeroModel(NativePolicy(), width=3, height=2)
    session = model.prepare_session(model.parameters)
    session.infer(observation(0))
    with pytest.raises(ValueError, match='strictly increasing'):
        session.infer(observation(0))


def test_droid_control_mode():
    assert recipe.DROID_CONTROL_MODE == command.to_wire(DROID_IMPEDANCE)


@pytest.mark.parametrize('mode', list(recipe.Mode))
def test_periodic_recipe_plays_commands_while_history_calls_return_none(mode):
    model = DreamZeroModel(NativePolicy(), width=3, height=2)
    session = model.prepare_session({
        **model.parameters,
        'mode': mode.value,
        'fps': 10,
        'sampling_period_sec': 0.1,
        'prediction_period_sec': 0.4,
    })
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)

    def round_trip(obs):
        return session.infer(serialization.deserialise(serialization.serialise(obs)))

    run = runtime.start(spec.from_spec(session.client_stack), round_trip)
    obs = {
        keys.JOINTS: np.zeros(7),
        keys.GRIP: 0.0,
        keys.TASK: 'Put banana in bowl.',
        **{source: np.zeros((2, 3, 3), np.uint8) for source in recipe.droid_cameras().values()},
    }
    try:
        run.send(obs)
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 50_000_000
        assert run.send(obs).commands[keys.ROBOT_COMMAND].positions[0] == 0
        for tick in (100_000_000, 200_000_000, 300_000_000):
            now[0] = tick
            run.send(obs)
            assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
            assert not run.send(obs).commands
        now[0] = 400_000_000
        run.send(obs)
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 420_000_000
        commands = run.send(obs).commands
        if mode is recipe.Mode.RTC:
            assert commands[keys.ROBOT_COMMAND].positions[0] == 1
        else:
            assert not commands
            now[0] = 450_000_000
            assert run.send(obs).commands[keys.ROBOT_COMMAND].positions[0] == 1
    finally:
        run.close()
        runtime.close()
        session.close()


def test_standalone_server_imports_without_positronic_or_native_model():
    script = """
import importlib.abc
import sys

class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'positronic', 'pimm', 'groot', 'torch'}:
            raise AssertionError(f'Standalone adapter imports {fullname}')

sys.meta_path.insert(0, ForbiddenImports())
sys.path.insert(0, sys.argv[1])
from serving import server
sys.argv = ['dreamzero_serving.server', 'droid', '--help']
server.main()
"""
    subprocess.run([sys.executable, '-I', '-c', script, str(Path(server.__file__).parents[1])], check=True)
