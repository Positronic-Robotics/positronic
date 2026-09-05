import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial

import numpy as np
import pos3
import pytest
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
from positronic.simulator.env_server.launcher import free_port, serve_subprocess
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


@pytest.mark.timeout(30.0)
def test_serve_subprocess_reports_a_server_that_dies_before_binding():
    """A server that raises during startup never binds, so its port stays closed exactly as a slow boot's
    does."""
    spawn = lambda host, port: subprocess.Popen([sys.executable, '-c', 'raise SystemExit(3)'])  # noqa: E731
    with pytest.raises(RuntimeError, match='status 3'), serve_subprocess(spawn, 'localhost'):
        pass


@pytest.mark.timeout(30.0)
def test_serve_subprocess_yields_once_the_port_accepts():
    script = (
        'import socket,sys,time\ns=socket.socket()\ns.bind(("localhost",int(sys.argv[1])))\ns.listen()\ntime.sleep(30)'
    )
    spawn = lambda host, port: subprocess.Popen([sys.executable, '-c', script, str(port)])  # noqa: E731
    with serve_subprocess(spawn, 'localhost') as (host, port):
        with socket.create_connection((host, port), timeout=1.0):
            pass


@pytest.mark.timeout(60.0)
def test_transport_is_transparent(env_server):
    """The wire round-trips faithfully: the same seed + action sequence through the env in-process and
    behind the socket produce identical raw observations. This is the parity oracle for the protocol."""
    host, port = env_server
    seed = 7

    direct = make_mujoco_env(list(CAMERAS.values()))
    direct_reset = direct.reset(seed)
    base = np.asarray(direct_reset[protocol.FRAME_OBS]['q'])
    actions = [
        {
            protocol.ACTION_COMMAND: {
                protocol.COMMAND_TYPE: protocol.JOINT_POS,
                protocol.COMMAND_JOINT_POS: base + 0.03 * i,
            },
            protocol.ACTION_GRIP: 0.2 * (i % 2),
        }
        for i in range(1, 6)
    ]
    direct_steps = [direct.step(action) for action in actions]
    direct.close()

    conn = EnvConnection(host, port)
    socket_reset = conn.reset(seed)
    socket_steps = [conn.step(action) for action in actions]
    conn.close()

    assert direct_reset[protocol.FRAME_CONTROL_DT] == socket_reset[protocol.FRAME_CONTROL_DT]
    _assert_obs_equal(direct_reset[protocol.FRAME_OBS], socket_reset[protocol.FRAME_OBS])
    for direct_step, socket_step in zip(direct_steps, socket_steps, strict=True):
        _assert_obs_equal(direct_step[protocol.FRAME_OBS], socket_step[protocol.FRAME_OBS])
        assert direct_step[protocol.FRAME_DONE] == socket_step[protocol.FRAME_DONE]
        assert direct_step[protocol.FRAME_CONTROL_DT] == socket_step[protocol.FRAME_CONTROL_DT]


@pytest.mark.timeout(60.0)
def test_the_env_answers_its_own_task_list(env_server):
    """The env answers its own records, and a spec it does not know raises through to the client."""
    host, port = env_server
    conn = EnvConnection(host, port)
    assert conn.tasks({}) == [{'name': SCENE_NAME}]
    assert conn.tasks({'task': SCENE_NAME}) == [{'name': SCENE_NAME}]
    with pytest.raises(RuntimeError, match='nosuchscene'):
        conn.tasks({'task': 'nosuchscene'})
    conn.close()


@contextmanager
def _mute_server():
    """A peer that accepts the connection and then answers nothing, holding the socket open.

    What a simulator looks like once it is wedged in its own teardown: the process is past serving but the
    socket outlives it, so nothing ever closes the connection from that end.
    """
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
    """Teardown ends the run: an unanswered goodbye is as good as a closed socket, and is not waited out."""
    with _mute_server() as (host, port):
        conn = EnvConnection(host, port)
        started = time.monotonic()
        conn.close()
        assert time.monotonic() - started < _CLOSE_ACK_TIMEOUT + 10.0


_HOLD = {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 0.0}


def _settle(env, action: dict, steps: int) -> np.ndarray:
    """Apply ``action`` once, then idle ``steps`` ticks while the position actuators settle; return the final eef."""
    out = env.step(action)
    for _ in range(steps):
        out = env.step(_HOLD)
    return np.asarray(out[protocol.FRAME_OBS]['ee_pos'])


def test_a_pinned_control_mode_rides_the_wire():
    """Honoring a mode is the env's, so the adapter delivers it rather than deciding for every env."""
    mode = roboarm_command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    wired = _wire_command(roboarm_command.JointPosition(np.zeros(7), mode=mode))
    assert wired[protocol.COMMAND_MODE] == roboarm_command.to_wire(mode)
    assert protocol.COMMAND_MODE not in _wire_command(roboarm_command.JointPosition(np.zeros(7)))


class TestEnvControlFrame:
    """An env measuring somewhere other than the embodiment's ``default`` gets commands re-expressed for it.

    ``RobolabAdapter`` is the live case, and this file covers the adapter wire contract, so its frame
    round-trip runs here.
    """

    frame = geom.Transform3D(np.array([0.0, 0.0, 0.1]), geom.Rotation.from_euler([0.0, 0.0, np.pi / 2]))
    rotmat = geom.Rotation.Representation.ROTATION_MATRIX

    def test_an_absolute_pose_arrives_in_the_env_frame(self):
        pose = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.from_euler([0.1, 0.2, 0.3]))
        wired = _wire_command(_in_env_control_frame(roboarm_command.CartesianPosition(pose), self.frame))
        np.testing.assert_allclose(wired[protocol.COMMAND_POSE], (pose * self.frame).as_vector(self.rotmat), atol=1e-12)

    def test_a_delta_already_in_the_env_frame_wires_bare(self):
        delta = geom.Transform3D(np.array([0.0, 0.0, 0.04]), geom.Rotation.identity)
        cmd = roboarm_command.CartesianDelta(delta, frame=self.frame)
        wired = _wire_command(_in_env_control_frame(cmd, self.frame))
        np.testing.assert_allclose(wired[protocol.COMMAND_DELTA], delta.as_vector(self.rotmat), atol=1e-12)

    def test_a_command_re_expressed_for_the_env_keeps_its_mode(self):
        """The frame a command is measured in has nothing to do with the law that drives to it."""
        mode = roboarm_command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
        pose = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.identity)
        moved = _in_env_control_frame(roboarm_command.CartesianPosition(pose, mode=mode), self.frame)
        assert moved.mode == mode
        delta = _in_env_control_frame(roboarm_command.CartesianDelta(pose, mode=mode), self.frame)
        assert delta.mode == mode

    def test_a_delta_outside_the_env_frame_is_refused(self):
        """The env anchors on its own measured pose, so a delta meant for another frame has no wire form."""
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
    """A CartesianDelta settles to the same eef as the absolute CartesianPosition for the composed target: it
    confirms the world-frame compose and that the delta fires once — a delta re-applied every tick would overshoot.
    Comparing the two paths cancels the actuators' shared steady-state offset, so the match is exact."""
    rotmat = geom.Rotation.Representation.ROTATION_MATRIX
    seed, settle = 11, 300
    lift = np.array([0.0, 0.0, 0.04])

    abs_env = make_mujoco_env(list(CAMERAS.values()))
    reset = abs_env.reset(seed)
    ee0 = np.asarray(reset[protocol.FRAME_OBS]['ee_pos'])
    target = geom.Transform3D(ee0 + lift, geom.Rotation.from_quat(reset[protocol.FRAME_OBS]['ee_quat']))
    absolute = {
        protocol.ACTION_COMMAND: {
            protocol.COMMAND_TYPE: protocol.CARTESIAN,
            protocol.COMMAND_POSE: target.as_vector(rotmat),
        },
        protocol.ACTION_GRIP: 0.0,
    }
    ee_abs = _settle(abs_env, absolute, settle)
    abs_env.close()

    delta_env = make_mujoco_env(list(CAMERAS.values()))
    delta_env.reset(seed)
    delta = geom.Transform3D(lift, geom.Rotation.identity)
    delta_action = {
        protocol.ACTION_COMMAND: {
            protocol.COMMAND_TYPE: protocol.CARTESIAN_DELTA,
            protocol.COMMAND_DELTA: delta.as_vector(rotmat),
        },
        protocol.ACTION_GRIP: 0.0,
    }
    ee_delta = _settle(delta_env, delta_action, settle)
    ee_idle = _settle(delta_env, _HOLD, 50)  # the delta already fired; idling must not re-compose it
    delta_env.close()

    assert ee_delta[2] > ee0[2] + 0.01, 'the delta did not lift the arm'
    np.testing.assert_allclose(ee_delta, ee_abs, atol=1e-4)  # same composed target -> same settled eef
    np.testing.assert_allclose(ee_idle, ee_delta, atol=1e-3)  # one-shot: idle ticks add no motion


# The one task ``_CountdownEnv`` serves.
_COUNTDOWN = 'countdown'


class _CountdownEnv(EnvProtocol):
    """A degenerate env exercising the proxy's terminal and free-run paths without the real ``stack_cubes``
    wrapper. Obs encodes the step count (``reset`` is step 0, each ``step`` increments) so a reader can
    tell whether the proxy stepped; ``done`` fires after ``done_after`` steps (``None`` → never)."""

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
        meta = {'task': _COUNTDOWN}  # scene meta the env reports only at reset; ``step`` omits it
        return {
            protocol.FRAME_OBS: {'q': np.full(7, self._steps, dtype=np.float64)},
            protocol.FRAME_META: meta,
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def step(self, action):
        self._steps += 1
        done = self._done_after is not None and self._steps >= self._done_after
        return {
            protocol.FRAME_OBS: {'q': np.full(7, self._steps, dtype=np.float64)},
            protocol.FRAME_DONE: done,
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

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
        return {eval_keys.SUCCESS: True} if result[protocol.FRAME_DONE] else None


@pytest.mark.timeout(60.0)
def test_the_proxy_connects_on_a_tasks_call_before_any_reset():
    """``tasks`` runs before the first trial, so it starts the server; the reset that follows shares its
    connection."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        obs_rx = world.pair(proxy.observations['value'])
        world.start([proxy])

        assert proxy.tasks({'task': _COUNTDOWN}) == [{eval_keys.TASK: _COUNTDOWN}]

        proxy.reset({eval_keys.SEED: 0})
        np.testing.assert_array_equal(obs_rx.value, np.zeros(7))


@pytest.mark.timeout(60.0)
def test_a_selection_naming_no_task_is_refused():
    """A sweep of no trials ends at once and reads as a run that succeeded, so an empty listing raises."""
    with serve_env(_CountdownEnv()) as (host, port):
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        with pytest.raises(ValueError, match='no task'):
            proxy.tasks({'task': []})


@pytest.mark.timeout(60.0)
def test_a_refused_listing_stops_the_server_it_started():
    """The scheduler enters ``run``, whose teardown stops the server, only after the listing."""
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
    """``reset`` publishes the env's frame (step 0) and clears ``done``, then the proxy free-runs — it steps
    the env every active tick (physics progresses through the inference window). The step-count obs makes it
    observable: the reset publishes step 0, then it advances each tick with no command needed."""
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
    """The env reports scene meta only at ``reset`` (``step`` omits it); the proxy caches it so a ``Task``
    reads its language live off ``proxy.meta`` — the callable-instruction path LIBERO relies on — and the
    cached value holds across the steps that follow."""
    with serve_env(_CountdownEnv()) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_CountdownAdapter(), nullcontext((host, port)))
        task = Task(instruction_source=lambda: proxy.meta['task'], timeout_sec=1.0)
        scheduler = world.start([proxy])

        proxy.reset({eval_keys.SEED: 0})
        assert task.instruction == 'countdown'  # resolved live off the cached reset meta
        drive_scheduler(scheduler, steps=4)  # the env steps, each ``step`` omitting meta ...
        assert task.instruction == 'countdown'  # ... yet the reset-scoped cache holds


def test_the_canonical_contract_is_exactly_what_a_client_can_emit():
    """``protocol`` owns the contract and ``_wire_command`` is the only thing that writes it, so pinning the two
    against each other keeps the set an env adoption must cover equal to the set a policy can actually emit."""
    pose = geom.Transform3D(np.zeros(3), geom.Rotation.identity)
    commands = [
        roboarm_command.CartesianPosition(pose),
        roboarm_command.CartesianDelta(pose),
        roboarm_command.JointPosition(np.zeros(7)),
        roboarm_command.JointDelta(np.zeros(7)),
        None,  # nothing held: the arm holds where it is
    ]
    tags = {_wire_command(command)[protocol.COMMAND_TYPE] for command in commands}
    assert tags == set(protocol.CANONICAL_COMMAND_TYPES)


@pytest.mark.timeout(60.0)
def test_remote_eval_runs_to_timeout_without_done(env_server, tmp_path):
    """The real ``stack_cubes`` wrapper, end to end: no terminal, so the trial runs to the task timeout
    (``eval.terminated`` False, ``eval.success`` absent) and records the canonical signals under the shared
    camera key."""
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
    """A codec names the camera it wants, so a sim spelling its cameras its own way can be scored only by a codec
    written for it. Every sim publishes the shared pair, whatever the benchmark calls those cameras."""
    observations = eval_cfg.instantiate().embodiment.observations
    assert {keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE} <= set(observations)


class _JointposChunks(Policy):
    """Chunks exactly as long as the intended open-loop cadence; ``target_grip`` encodes
    ``chunk * 100 + step`` so the recorded wire signals show which actions executed, and when."""

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
    """The recording proves the contract the DROID jointpos codec makes with RoboLab's client: every action
    of every chunk lands on the wire — including the final one, which ``ActionTimestamp``'s validity
    sentinel gives a full period before ``ChunkedSchedule`` re-infers — and replans arrive exactly
    ``chunk_len`` control periods apart."""
    host, port = env_server
    probe = make_mujoco_env([])
    control_dt = probe.reset(0)[protocol.FRAME_CONTROL_DT]
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
def test_server_failure_crosses_as_error_frame(env_server):
    """A command the env rejects comes back as an error the client re-raises — the connection survives
    rather than dying on the server-side exception, and the next command still works."""
    host, port = env_server
    conn = EnvConnection(host, port)
    conn.reset(7)
    with pytest.raises(RuntimeError, match='bogus'):
        conn.step({protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: 'bogus'}, protocol.ACTION_GRIP: 0.0})
    # The socket is still usable after a delivered failure.
    joints = {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: np.zeros(7)}
    step = conn.step({protocol.ACTION_COMMAND: joints, protocol.ACTION_GRIP: 0.0})
    assert protocol.FRAME_OBS in step
    conn.close()
