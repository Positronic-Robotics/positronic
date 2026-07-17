from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pos3
import pytest

import pimm
from positronic import geom
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import Task
from positronic.inference import main
from positronic.policy.tests.test_harness import StubPolicy
from positronic.policy.wrappers import ChunkedSchedule
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.env_server.server import EnvProtocol
from positronic.simulator.env_server.tests.conftest import serve_env
from positronic.simulator.env_server.tests.mujoco_env import CAMERAS, make_mujoco_env, remote_stack_cubes_eval
from positronic.tests.testing_coutils import drive_scheduler


class FakeRenderer:
    """Stand-in for ``mj.Renderer`` so the server (in a thread here) never touches a GL context."""

    def __init__(self, _model, *, height, width, max_geom=10000, font_scale=None):
        self.height = height
        self.width = width

    def update_scene(self, _data, camera=None):
        pass

    def render(self, out=None):
        if out is not None:
            out[:] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return None
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _fake_renderer(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    monkeypatch.setattr('positronic.simulator.mujoco.sim.mj.Renderer', FakeRenderer)


def _assert_obs_equal(a: dict, b: dict) -> None:
    for key in ('q', 'dq', 'ee_pos', 'ee_quat'):
        np.testing.assert_array_equal(a[key], b[key])
    assert a['status'] == b['status']
    assert a['grip'] == b['grip']
    assert a['cameras'].keys() == b['cameras'].keys()
    for name in a['cameras']:
        np.testing.assert_array_equal(a['cameras'][name], b['cameras'][name])
    assert a['sim_state'].keys() == b['sim_state'].keys()
    for key in a['sim_state']:
        np.testing.assert_array_equal(a['sim_state'][key], b['sim_state'][key])


@pytest.mark.timeout(60.0)
def test_transport_is_transparent(env_server):
    """The wire round-trips faithfully: the same seed + action sequence through the env in-process and
    behind the socket produce identical raw observations. This is the parity oracle for the protocol."""
    host, port = env_server
    seed = 7

    direct = make_mujoco_env(list(CAMERAS.values()))
    direct_reset = direct.reset(seed)
    base = np.asarray(direct_reset['obs']['q'])
    actions = [{'joints': base + 0.03 * i, 'grip': 0.2 * (i % 2)} for i in range(1, 6)]
    direct_steps = [direct.step(action) for action in actions]
    direct.close()

    conn = EnvConnection(host, port)
    socket_reset = conn.reset(seed)
    socket_steps = [conn.step(action) for action in actions]
    conn.close()

    assert direct_reset['control_dt'] == socket_reset['control_dt']
    _assert_obs_equal(direct_reset['obs'], socket_reset['obs'])
    for direct_step, socket_step in zip(direct_steps, socket_steps, strict=True):
        _assert_obs_equal(direct_step['obs'], socket_step['obs'])
        assert direct_step['done'] == socket_step['done']
        assert direct_step['control_dt'] == socket_step['control_dt']


def _settle(env, action: dict, steps: int) -> np.ndarray:
    """Apply ``action`` once, then idle ``steps`` ticks while the position actuators settle; return the final eef."""
    env.step(action)
    out = {'obs': None}
    for _ in range(steps):
        out = env.step({})
    return np.asarray(out['obs']['ee_pos'])


@pytest.mark.timeout(60.0)
def test_cartesian_delta_matches_absolute_target():
    """A CartesianDelta settles to the same eef as the absolute CartesianPosition for the composed target: it
    confirms the world-frame compose and that the delta fires once — a delta re-applied every tick would overshoot.
    Comparing the two paths cancels the actuators' shared steady-state offset, so the match is exact."""
    rotmat = geom.Rotation.Representation.ROTATION_MATRIX
    seed, settle = 11, 300
    lift = np.array([0.0, 0.0, 0.04])

    abs_env = make_mujoco_env(list(CAMERAS.values()))
    reset = abs_env.reset(seed)
    ee0 = np.asarray(reset['obs']['ee_pos'])
    target = geom.Transform3D(ee0 + lift, geom.Rotation.from_quat(reset['obs']['ee_quat']))
    ee_abs = _settle(abs_env, {'pose': target.as_vector(rotmat)}, settle)
    abs_env.close()

    delta_env = make_mujoco_env(list(CAMERAS.values()))
    delta_env.reset(seed)
    delta = geom.Transform3D(lift, geom.Rotation.identity)
    ee_delta = _settle(delta_env, {'pose_delta': delta.as_vector(rotmat)}, settle)
    ee_idle = _settle(delta_env, {}, 50)  # the delta already fired; idling must not re-compose it
    delta_env.close()

    assert ee_delta[2] > ee0[2] + 0.01, 'the delta did not lift the arm'
    np.testing.assert_allclose(ee_delta, ee_abs, atol=1e-4)  # same composed target -> same settled eef
    np.testing.assert_allclose(ee_idle, ee_delta, atol=1e-3)  # one-shot: idle ticks add no motion


class _CountdownEnv(EnvProtocol):
    """A degenerate env exercising the proxy's terminal and free-run paths without the real ``stack_cubes``
    wrapper. Obs encodes the step count (``reset`` is step 0, each ``step`` increments) so a reader can
    tell whether the proxy stepped; ``done`` fires after ``done_after`` steps (``None`` → never)."""

    def __init__(self, done_after: int | None = None, control_dt: float = 0.1):
        self._done_after = done_after
        self._control_dt = control_dt
        self._steps = 0

    def reset(self, token):
        self._steps = 0
        meta = {'task': 'countdown'}  # scene meta the env reports only at reset; ``step`` omits it
        return {
            'obs': {'q': np.full(7, self._steps, dtype=np.float64)},
            'meta': meta,
            'robot_meta': {},
            'control_dt': self._control_dt,
        }

    def step(self, action):
        self._steps += 1
        done = self._done_after is not None and self._steps >= self._done_after
        return {'obs': {'q': np.full(7, self._steps, dtype=np.float64)}, 'done': done, 'control_dt': self._control_dt}

    def close(self):
        pass


class _CountdownAdapter(EnvAdapter):
    def reset_token(self, context):
        return context.get('eval.seed')

    def action(self, commands, now_ns):
        return {}

    def observations(self, raw_obs):
        return {'value': raw_obs['q']}

    def privileged(self, raw_obs):
        return {}

    def terminal(self, result):
        return {'eval.success': True} if result['done'] else None


@pytest.mark.timeout(60.0)
def test_proxy_publishes_frame0_then_free_runs():
    """``reset`` arms frame-0 (step 0); the proxy publishes it on its next turn and clears ``done``, then
    free-runs — it steps the env every active tick (physics progresses through the inference window). The
    step-count obs makes it observable: frame-0 is step 0, then it advances each tick with no command needed."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        obs_rx = world.pair(proxy.observations['value'])
        done_rx = world.pair(proxy.done)

        scheduler = world.start([proxy])
        drive_scheduler(scheduler, steps=2)  # inactive: the proxy paces time without an env

        proxy.reset({'eval.seed': 0})  # arm frame-0; the run loop publishes it on its next turn
        drive_scheduler(scheduler, steps=1)
        np.testing.assert_array_equal(obs_rx.read().data, np.zeros(7))
        assert done_rx.read().data == {}

        drive_scheduler(scheduler, steps=3)  # free-run: the env steps even with no command delivered
        assert obs_rx.read().data[0] >= 1


@pytest.mark.timeout(60.0)
def test_proxy_caches_reset_meta_as_live_instruction_source():
    """The env reports scene meta only at ``reset`` (``step`` omits it); the proxy caches it so a ``Task``
    reads its language live off ``proxy.meta`` — the callable-instruction path LIBERO relies on — and the
    cached value holds across the steps that follow."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        task = Task(instruction=lambda: proxy.meta['task'], timeout=1.0, reset=proxy.reset)
        scheduler = world.start([proxy])

        proxy.reset({'eval.seed': 0})
        assert task.instruction == 'countdown'  # resolved live off the cached reset meta
        drive_scheduler(scheduler, steps=4)  # the env steps, each ``step`` omitting meta ...
        assert task.instruction == 'countdown'  # ... yet the reset-scoped cache holds


@pytest.mark.timeout(60.0)
def test_remote_eval_runs_to_timeout_without_done(env_server, tmp_path):
    """The real ``stack_cubes`` wrapper, end to end: no terminal, so the trial runs to the task timeout
    (``eval.terminated`` False) and records the canonical signals."""
    host, port = env_server
    with pos3.mirror():
        ev = remote_stack_cubes_eval(host, port, camera_dict=CAMERAS)
        ev.task.timeout = 0.1
        policy = StubPolicy(command=roboarm_command.JointPosition(np.zeros(7)), target_grip=0.0)
        main(
            policy=policy,
            evals=[replace(ev, trials=[{'eval.trial_index': 0, 'eval.seed': 100}])],
            output_dir=str(tmp_path),
            wrap=ChunkedSchedule(),
        )

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    episode = ds[0]
    assert episode.static['eval.terminated'] is False
    assert episode.static['eval.universe'] == 'sim'
    assert episode.static['eval.embodiment'] == 'remote.mujoco.franka'
    assert episode.static['scene_xml'].startswith('<mujoco')
    signals = episode.signals
    assert 'image.agentview' in signals
    assert 'robot_command.joints' in signals
    assert 'sim_state.mjSTATE_INTEGRATION' in signals


@pytest.mark.timeout(60.0)
def test_server_failure_crosses_as_error_frame(env_server):
    """A command the env rejects comes back as an error the client re-raises — the connection survives
    rather than dying on the server-side exception, and the next command still works."""
    host, port = env_server
    conn = EnvConnection(host, port)
    conn.reset(7)
    with pytest.raises(RuntimeError, match='bogus'):
        conn.step({'bogus': True})
    assert 'obs' in conn.step({'joints': np.zeros(7)})  # the socket is still usable after a delivered failure
    conn.close()
