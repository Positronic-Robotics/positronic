import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial

import numpy as np
import pos3
import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import OP_PING
from websockets.sync.client import connect as websocket_connect
from websockets.sync.server import serve as websocket_serve

import pimm
from positronic import geom, keys
from positronic.cfg.eval import number_trials
from positronic.cfg.eval.sim import libero as libero_cfg
from positronic.cfg.eval.sim import positronic as native_cfg
from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.cli.eval.run import main
from positronic.dataset import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.policy import Policy, Session
from positronic.policy.codec import ActionTimestamp
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.tests.test_harness import StubPolicy
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import EnvAdapter, _in_env_control_frame, _wire_command
from positronic.simulator.env_server.client import _CLOSE_ACK_TIMEOUT, EnvConnection
from positronic.simulator.env_server.launcher import free_port
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.env_server.server import EnvProtocol
from positronic.simulator.env_server.tests.conftest import serve_env
from positronic.simulator.env_server.tests.mujoco_env import (
    CAMERAS,
    SCENE_NAME,
    make_mujoco_env,
    remote_stack_cubes_eval,
)
from positronic.simulator.robolab.adapter import RobolabAdapter
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
    """The same seed and actions must yield identical raw observations in-process and over the socket."""
    host, port = env_server
    seed = 7

    direct = make_mujoco_env(list(CAMERAS.values()))
    direct_reset = direct.reset(seed)
    base = np.asarray(direct_reset['obs']['q'])
    actions = [{'command': {'type': 'joint_pos', 'q': base + 0.03 * i}, 'grip': 0.2 * (i % 2)} for i in range(1, 6)]
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


@pytest.mark.timeout(60.0)
def test_the_env_answers_its_own_task_list(env_server):
    """Unknown task specifications must reach the environment and return its errors."""
    host, port = env_server
    conn = EnvConnection(host, port)
    assert conn.tasks({}) == [{'name': SCENE_NAME}]
    assert conn.tasks({'task': SCENE_NAME}) == [{'name': SCENE_NAME}]
    with pytest.raises(RuntimeError, match='nosuchscene'):
        conn.tasks({'task': 'nosuchscene'})
    conn.close()


@contextmanager
def _mute_server():
    """Hold the socket open without answering requests, simulating a peer stuck in teardown."""
    host, port = 'localhost', free_port()
    release = threading.Event()

    def handler(connection):
        for _ in connection:
            release.wait(timeout=60.0)

    server = websocket_serve(handler, host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield host, port
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5.0)


@pytest.mark.timeout(60.0)
def test_close_gives_up_on_a_peer_that_never_answers():
    with _mute_server() as (host, port):
        conn = EnvConnection(host, port)
        started = time.monotonic()
        conn.close()
        assert time.monotonic() - started < _CLOSE_ACK_TIMEOUT + 10.0


@pytest.fixture
def server_without_heartbeat(monkeypatch):
    monkeypatch.setattr(
        'positronic.simulator.env_server.client.connect',
        partial(websocket_connect, ping_interval=0.01, ping_timeout=0.02),
    )
    ignored_pings = []
    release = threading.Event()

    def handler(connection):
        receive_frame = connection.protocol.recv_frame

        def ignore_ping(frame):
            if frame.opcode == OP_PING:
                ignored_pings.append(frame)
            else:
                receive_frame(frame)

        monkeypatch.setattr(connection.protocol, 'recv_frame', ignore_ping)
        for raw in connection:
            if protocol.Command(protocol.decode(raw)[protocol.CMD]) is protocol.Command.CLOSE:
                connection.send(protocol.encode({protocol.OK: True}))
                return
            if release.wait(timeout=0.2):
                return
            connection.send(protocol.encode({'obs': {'ready': True}}))

    host, port = 'localhost', free_port()
    with websocket_serve(handler, host, port) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield host, port, ignored_pings
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=2.0)


@pytest.mark.timeout(10.0)
def test_scene_reset_survives_delayed_heartbeat_replies(server_without_heartbeat):
    host, port, ignored_pings = server_without_heartbeat
    conn = EnvConnection(host, port)
    try:
        assert conn.reset({}) == {'obs': {'ready': True}}
        assert ignored_pings
    finally:
        conn.close()


@pytest.mark.timeout(10.0)
@pytest.mark.parametrize('command', [EnvConnection.tasks, EnvConnection.reset, EnvConnection.step])
def test_unanswered_heartbeat_closes_pending_requests(server_without_heartbeat, command):
    host, port, ignored_pings = server_without_heartbeat
    conn = EnvConnection(host, port, ping_timeout=0.02)
    try:
        with pytest.raises(ConnectionClosedError, match='keepalive ping timeout'):
            command(conn, {})
        assert ignored_pings
        with pytest.raises(ConnectionClosedError):
            conn.step({})
    finally:
        conn.close()


_HOLD = {'command': {'type': 'hold'}, 'grip': 0.0}


def _settle(env, action: dict, steps: int) -> np.ndarray:
    """Apply the action once, hold for ``steps`` ticks, and return the settled end-effector position."""
    env.step(action)
    out = {'obs': None}
    for _ in range(steps):
        out = env.step(_HOLD)
    return np.asarray(out['obs']['ee_pos'])


def test_a_pinned_control_mode_rides_the_wire():
    """Control modes must pass through for the environment to interpret."""
    mode = roboarm_command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    wired = _wire_command(roboarm_command.JointPosition(np.zeros(7), mode=mode))
    assert wired['mode'] == roboarm_command.to_wire(mode)
    assert 'mode' not in _wire_command(roboarm_command.JointPosition(np.zeros(7)))


class TestEnvControlFrame:
    """Commands must target the frame the environment measures, even when it differs from the default."""

    frame = geom.Transform3D(np.array([0.0, 0.0, 0.1]), geom.Rotation.from_euler([0.0, 0.0, np.pi / 2]))
    rotmat = geom.Rotation.Representation.ROTATION_MATRIX

    def test_an_absolute_pose_arrives_in_the_env_frame(self):
        pose = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.from_euler([0.1, 0.2, 0.3]))
        wired = _wire_command(_in_env_control_frame(roboarm_command.CartesianPosition(pose), self.frame))
        np.testing.assert_allclose(wired['pose'], (pose * self.frame).as_vector(self.rotmat), atol=1e-12)

    def test_a_delta_already_in_the_env_frame_wires_bare(self):
        delta = geom.Transform3D(np.array([0.0, 0.0, 0.04]), geom.Rotation.identity)
        cmd = roboarm_command.CartesianDelta(delta, frame=self.frame)
        wired = _wire_command(_in_env_control_frame(cmd, self.frame))
        np.testing.assert_allclose(wired['delta'], delta.as_vector(self.rotmat), atol=1e-12)

    def test_a_command_re_expressed_for_the_env_keeps_its_mode(self):
        mode = roboarm_command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
        pose = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.identity)
        moved = _in_env_control_frame(roboarm_command.CartesianPosition(pose, mode=mode), self.frame)
        assert moved.mode == mode
        delta = _in_env_control_frame(roboarm_command.CartesianDelta(pose, mode=mode), self.frame)
        assert delta.mode == mode

    def test_a_delta_outside_the_env_frame_is_refused(self):
        """A delta needs the measured pose in its own frame, which the wire does not supply."""
        delta = geom.Transform3D(np.array([0.0, 0.0, 0.04]), geom.Rotation.identity)
        with pytest.raises(ValueError, match='control frame'):
            _wire_command(_in_env_control_frame(roboarm_command.CartesianDelta(delta), self.frame))

    def test_robolab_reports_and_drives_the_same_frame(self):
        adapter = RobolabAdapter(camera_dict={})
        eef = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.from_euler([0.1, 0.2, 0.3]))
        raw = {
            'eef_pos': eef.translation,
            'eef_quat': eef.rotation.as_quat,
            'joint_pos': np.zeros(7),
            'joint_vel': np.zeros(7),
            'grip': 0.0,
        }
        reported = adapter.observations(raw)[keys.ROBOT_STATE].ee_pose
        commanded = _in_env_control_frame(roboarm_command.CartesianPosition(reported), adapter.env_control_frame).pose
        np.testing.assert_allclose(commanded.as_vector(self.rotmat), eef.as_vector(self.rotmat), atol=1e-6)


@pytest.mark.timeout(60.0)
def test_cartesian_delta_matches_absolute_target():
    """A one-shot delta must settle at the composed absolute target without accumulating on idle ticks.

    Comparing both paths cancels their shared actuator steady-state offset.
    """
    rotmat = geom.Rotation.Representation.ROTATION_MATRIX
    seed, settle = 11, 300
    lift = np.array([0.0, 0.0, 0.04])

    abs_env = make_mujoco_env(list(CAMERAS.values()))
    reset = abs_env.reset(seed)
    ee0 = np.asarray(reset['obs']['ee_pos'])
    target = geom.Transform3D(ee0 + lift, geom.Rotation.from_quat(reset['obs']['ee_quat']))
    ee_abs = _settle(abs_env, {'command': {'type': 'cartesian', 'pose': target.as_vector(rotmat)}, 'grip': 0.0}, settle)
    abs_env.close()

    delta_env = make_mujoco_env(list(CAMERAS.values()))
    delta_env.reset(seed)
    delta = geom.Transform3D(lift, geom.Rotation.identity)
    delta_action = {'command': {'type': 'cartesian_delta', 'delta': delta.as_vector(rotmat)}, 'grip': 0.0}
    ee_delta = _settle(delta_env, delta_action, settle)
    ee_idle = _settle(delta_env, _HOLD, 50)  # the delta already fired; idling must not re-compose it
    delta_env.close()

    assert ee_delta[2] > ee0[2] + 0.01, 'the delta did not lift the arm'
    np.testing.assert_allclose(ee_delta, ee_abs, atol=1e-4)
    np.testing.assert_allclose(ee_idle, ee_delta, atol=1e-3)


_COUNTDOWN = 'countdown'


class _CountdownEnv(EnvProtocol):
    """Observe step counts starting at zero on reset; ``done_after=None`` never terminates."""

    def __init__(self, done_after: int | None = None, control_dt: float = 0.1):
        self._done_after = done_after
        self._control_dt = control_dt
        self._steps = 0

    def tasks(self, spec):
        selection = spec.get('task', _COUNTDOWN)
        names = [selection] if isinstance(selection, str) else list(selection)
        if any(name != _COUNTDOWN for name in names):
            raise ValueError(f'_CountdownEnv serves {_COUNTDOWN!r}, not {names}')
        return [{'name': name} for name in names]

    def reset(self, token):
        self._steps = 0
        meta = {'task': _COUNTDOWN}
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
    def task_params(self, records):
        return [{eval_keys.TASK: record['name']} for record in records]

    def reset_token(self, params):
        return params.get(eval_keys.SEED)

    def action(self, commands):
        return {}

    def observations(self, raw_obs):
        return {'value': raw_obs['q']}

    def privileged(self, raw_obs):
        return {}

    def terminal(self, result):
        return {eval_keys.SUCCESS: True} if result['done'] else None


@pytest.mark.timeout(60.0)
def test_the_proxy_connects_on_a_tasks_call_before_any_reset():
    """Task listing must start the server and leave the connection usable for reset."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        obs_rx = world.pair(proxy.observations['value'])
        world.start([proxy])

        assert proxy.tasks({'task': _COUNTDOWN}) == [{eval_keys.TASK: _COUNTDOWN}]

        proxy.reset({eval_keys.SEED: 0})
        np.testing.assert_array_equal(obs_rx.value, np.zeros(7))


@pytest.mark.timeout(60.0)
def test_a_selection_naming_no_task_is_refused():
    """Reject empty task selections so zero-trial runs cannot appear successful."""
    with serve_env(_CountdownEnv()) as (host, port):
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        with pytest.raises(ValueError, match='no task'):
            proxy.tasks({'task': []})


@pytest.mark.timeout(60.0)
def test_a_refused_listing_stops_the_server_it_started():
    """Listing can fail before the scheduler starts, so cleanup cannot depend on its teardown."""
    with serve_env(_CountdownEnv()) as address:
        stopped = False

        @contextmanager
        def serve():
            nonlocal stopped
            try:
                yield address
            finally:
                stopped = True

        proxy = RemoteEnvControlSystem(_CountdownAdapter(), serve())
        with pytest.raises(RuntimeError, match='nosuchtask'):
            proxy.tasks({'task': 'nosuchtask'})
        assert stopped, 'the server stayed up with nobody left to stop it'


@pytest.mark.timeout(60.0)
def test_proxy_publishes_the_reset_frame_then_free_runs():
    """Reset must publish step zero and clear termination; active ticks advance physics without commands."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        obs_rx = world.pair(proxy.observations['value'])
        done_rx = world.pair(proxy.done)

        scheduler = world.start([proxy])
        drive_scheduler(scheduler, steps=2)  # inactive: the proxy paces time without an env

        proxy.reset({eval_keys.SEED: 0})
        np.testing.assert_array_equal(obs_rx.read().data, np.zeros(7))
        assert done_rx.read().data == {}

        drive_scheduler(scheduler, steps=3)  # free-run: the env steps even with no command delivered
        assert obs_rx.read().data[0] >= 1


@pytest.mark.timeout(60.0)
def test_proxy_caches_reset_meta_as_live_instruction_source():
    """Live instruction callbacks must retain reset metadata across steps that omit it."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        task = Task(instruction_source=lambda: proxy.meta['task'], timeout_sec=1.0)
        scheduler = world.start([proxy])

        proxy.reset({eval_keys.SEED: 0})
        assert task.instruction == 'countdown'
        drive_scheduler(scheduler, steps=4)
        assert task.instruction == 'countdown'


@pytest.mark.timeout(60.0)
def test_remote_eval_runs_to_timeout_without_done(env_server, tmp_path):
    """A timed-out trial must record canonical signals without reporting termination or success."""
    host, port = env_server
    with pos3.mirror():
        ev = remote_stack_cubes_eval(host, port, camera_dict=CAMERAS)
        trial = number_trials([(replace(next(iter(ev.tasks())), timeout_sec=0.1), {eval_keys.SEED: 100})])[0]
        policy = StubPolicy(command=roboarm_command.JointPosition(np.zeros(7)), target_grip=0.0)
        main(
            policy=ChunkedSchedule().wrap(policy),
            evals=[replace(ev, tasks=partial(iter, [trial]))],
            output_dir=str(tmp_path),
        )

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    episode = ds[0]
    assert isinstance(episode, Episode)
    assert episode.static[eval_keys.TERMINATED] is False
    assert eval_keys.SUCCESS not in episode.static
    assert episode.static[eval_keys.UNIVERSE] == 'sim'
    assert episode.static[eval_keys.EMBODIMENT] == 'remote.mujoco.franka'
    assert episode.static['scene_xml'].startswith('<mujoco')
    signals = episode.signals
    assert keys.EXTERIOR_IMAGE in signals
    assert keys.TARGET_JOINTS in signals
    assert 'sim_state.mjSTATE_INTEGRATION' in signals


@pytest.mark.parametrize(
    'eval_cfg', [libero_cfg.spatial, robolab_cfg.benchmark, native_cfg.stack_cubes], ids=['libero', 'robolab', 'mujoco']
)
def test_every_sim_eval_publishes_the_shared_camera_keys(eval_cfg):
    """Shared camera names let the same policy codec serve different simulators."""
    observations = eval_cfg.instantiate().embodiment.observations
    assert {keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE} <= set(observations)


class _JointposChunks(Policy):
    """Encode ``chunk * 100 + step`` in grip values to identify executed actions in recordings."""

    def __init__(self, command: roboarm_command.CommandType, chunk_len: int):
        self.command = command
        self.chunk_len = chunk_len
        self.chunks = 0

    def new_session(self, context=None, rt=None):
        return _JointposChunkSession(self)


class _JointposChunkSession(Session):
    def __init__(self, policy: _JointposChunks):
        self._policy = policy

    def __call__(self, obs, time_ns):
        self._policy.chunks += 1
        return [
            {keys.ROBOT_COMMAND: self._policy.command, 'target_grip': self._policy.chunks * 100.0 + i}
            for i in range(self._policy.chunk_len)
        ]


@pytest.mark.timeout(60.0)
def test_full_chunk_executes_between_replans(env_server, tmp_path):
    """Every chunk action must execute, with a full control period for the final action.

    ``ActionTimestamp``'s validity sentinel must keep replans ``chunk_len`` control periods apart.
    """
    host, port = env_server
    probe = make_mujoco_env([])
    control_dt = probe.reset(0)['control_dt']
    probe.close()

    chunk_len = 5
    raw = _JointposChunks(roboarm_command.JointPosition(np.zeros(7)), chunk_len)
    policy = (ChunkedSchedule() | ActionTimestamp(fps=1.0 / control_dt)).wrap(raw)
    with pos3.mirror():
        ev = remote_stack_cubes_eval(host, port, camera_dict=CAMERAS)
        task = replace(next(iter(ev.tasks())), timeout_sec=20 * control_dt)
        trial = number_trials([(task, {eval_keys.SEED: 100})])[0]
        main(policy=policy, evals=[replace(ev, tasks=partial(iter, [trial]))], output_dir=str(tmp_path))

    grip = LocalDataset(tmp_path)[0].signals['target_grip']
    executed = [(float(v), int(ts)) for v, ts in (grip[i] for i in range(len(grip)))]
    values = [v for v, _ in executed]  # every sample is a chunk action: nothing else commands this channel
    complete_chunks = raw.chunks - 1  # the deadline cuts the last chunk short
    assert complete_chunks >= 2
    expected = [c * 100.0 + i for c in range(1, complete_chunks + 1) for i in range(chunk_len)]
    assert values[: len(expected)] == expected

    starts = [ts for v, ts in executed if v % 100 == 0]
    period_ns = chunk_len * control_dt * 1e9
    for earlier, later in zip(starts, starts[1:], strict=False):
        assert later - earlier == pytest.approx(period_ns, abs=period_ns / (2 * chunk_len))


@pytest.mark.timeout(60.0)
@pytest.mark.parametrize(
    'message',
    [
        {protocol.CMD: 'bogus'},
        {protocol.CMD: protocol.Command.STEP.value, protocol.ACTION: {'command': {'type': 'bogus'}, 'grip': 0.0}},
    ],
)
def test_server_failure_crosses_as_error_frame(env_server, message):
    """Rejected commands must reach the client as errors while leaving the connection usable."""
    host, port = env_server
    conn = EnvConnection(host, port)
    conn.reset(7)
    with pytest.raises(RuntimeError, match='bogus'):
        conn._request(message)
    assert 'obs' in conn.step({'command': {'type': 'joint_pos', 'q': np.zeros(7)}, 'grip': 0.0})
    conn.close()
