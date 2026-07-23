"""The native test fixture: ``MujocoSim``/``stack_cubes`` wrapped behind the env-server protocol.

This is the duplicated, test-only remote copy of the production native path (in-process
``stack_cubes`` stays the real one), and the reference ``WireCommandAdapter`` env — it decodes the
same tagged command dialect the real benchmarks speak, with no OSC/axis-angle quirks. ``MujocoEnv``
drives ``MujocoSim.run()`` directly, binding local pipes to its ports the way ``World.start`` does,
and exposes it as the gym-style ``reset``/``step`` the server serves. ``make_mujoco_env`` builds it;
``StackCubesAdapter`` is the client-side mapping.
"""

from collections import deque
from contextlib import nullcontext
from typing import Any

import numpy as np

import pimm
import positronic.cfg.simulator
from pimm.world import LocalQueueEmitter, LocalQueueReceiver, VirtualClock
from positronic import geom, keys
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import Eval, Observation, Task
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.env_server.server import EnvProtocol
from positronic.simulator.mujoco.sim import MujocoFrankaState, MujocoSim
from positronic.utils import package_assets_path

_ROTMAT = geom.Rotation.Representation.ROTATION_MATRIX

# Logical observation name -> the model camera the sim renders.
CAMERAS = {'image.agentview': 'agentview'}


class _NeverStop(pimm.SignalReceiver):
    """A ``should_stop`` that never fires — ``MujocoEnv`` drives the sim loop tick by tick instead."""

    def read(self) -> pimm.Message:
        return pimm.Message(False, 0, False)


class MujocoEnv(EnvProtocol):
    """A ``MujocoSim`` behind the gym-style env protocol, driving its ``run()`` loop one tick at a time.

    Binding local pipes to the sim's ports (the wiring ``World.start`` performs) lets ``step`` inject a
    raw action and pump the fused per-tick loop, then read the post-step signals back. The control
    period is the sim's own physics ``timestep`` — the env has no other dt — reported from ``reset``.
    """

    def __init__(self, sim: MujocoSim, camera_names: list[str]):
        self._sim = sim
        self._clock = VirtualClock()
        self._gen = None
        self._cmd_emit = self._bind_input(sim.commands)
        self._grip_emit = self._bind_input(sim.target_grip)
        self._state_recv = self._bind_output(sim.state)
        self._grip_recv = self._bind_output(sim.grip)
        self._robot_meta_recv = self._bind_output(sim.robot_meta)
        self._sim_state_recv = self._bind_output(sim.sim_state)
        self._camera_recvs = {name: self._bind_output(sim.cameras[name]) for name in camera_names}

    def _bind_input(self, receiver: pimm.ControlSystemReceiver) -> LocalQueueEmitter:
        queue: deque = deque(maxlen=1)
        receiver._bind(LocalQueueReceiver(queue))
        return LocalQueueEmitter(queue, self._clock)

    def _bind_output(self, emitter: pimm.ControlSystemEmitter) -> LocalQueueReceiver:
        queue: deque = deque(maxlen=1)
        emitter._bind(LocalQueueEmitter(queue, self._clock))
        return LocalQueueReceiver(queue)

    @property
    def _timestep(self) -> float:
        return self._sim.model.opt.timestep

    def _advance(self, duration: float) -> None:
        target_ns = self._clock.now_ns() + max(1, round(duration * 1e9))
        while self._clock.now_ns() < target_ns:
            command = next(self._gen)
            step_ns = max(1, round(command.seconds * 1e9)) if isinstance(command, pimm.Sleep) else 1
            self._clock.advance_to_ns(self._clock.now_ns() + step_ns)

    def _read_obs(self) -> dict[str, Any]:
        state = self._state_recv.read().data
        ee_pose = state.ee_pose
        # ``q``/``dq`` already copy out of the sim's reused state buffer; ``ee_pose`` translation is a
        # live view into it, so copy here too — otherwise the next ``step`` overwrites this obs.
        return {
            'q': state.q,
            'dq': state.dq,
            'ee_pos': np.array(ee_pose.translation),
            'ee_quat': np.array(ee_pose.rotation.as_quat),
            'status': int(state.status.value),
            'grip': float(self._grip_recv.read().data),
            'cameras': {name: recv.read().data.array.copy() for name, recv in self._camera_recvs.items()},
            'sim_state': dict(self._sim_state_recv.read().data),
        }

    def reset(self, token: Any) -> dict[str, Any]:
        # Close the prior generator before rebuilding the scene, so its ``run`` cleanup tears down the old
        # renderer before ``reset`` creates the new one. ``reset`` loads the scene and arms frame-0; pump the
        # run loop twice — its setup + first control-period sleep, then the reset turn that publishes frame-0 —
        # and return that frame without stepping. The first ``step`` advances the generator with the first
        # action (Gym-style).
        if self._gen is not None:
            self._gen.close()
        self._sim.reset(token)
        self._gen = self._sim.run(_NeverStop(), self._clock)
        next(self._gen)  # loop setup + first control-period sleep
        next(self._gen)  # the reset turn: publishes frame-0 (no step)
        # Native ``stack_cubes`` has no language scene meta (its instruction is a static client string), so ``meta``
        # is empty; the sim's robot identity (URDF / joints) is the ``robot_meta``.
        return {
            'obs': self._read_obs(),
            'meta': {},
            'robot_meta': dict(self._robot_meta_recv.read().data),
            'control_dt': self._timestep,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        assert self._gen is not None, 'step() called before reset()'  # real Gym envs reject step-before-reset
        command = action['command']
        match command['type']:
            case 'hold':
                pass
            case 'joint_pos':
                self._cmd_emit.emit(roboarm_command.JointPosition(np.asarray(command['q'], dtype=np.float64)))
            case 'joint_vel':
                self._cmd_emit.emit(roboarm_command.JointDelta(np.asarray(command['dq'], dtype=np.float64)))
            case 'cartesian':
                self._cmd_emit.emit(
                    roboarm_command.CartesianPosition(geom.Transform3D.from_vector(command['pose'], _ROTMAT))
                )
            case 'cartesian_delta':
                self._cmd_emit.emit(
                    roboarm_command.CartesianDelta(geom.Transform3D.from_vector(command['delta'], _ROTMAT))
                )
            case other:
                raise ValueError(f'MujocoEnv got unsupported command type {other!r}')
        self._grip_emit.emit(float(action['grip']))
        self._advance(self._timestep)
        return {'obs': self._read_obs(), 'done': False, 'control_dt': self._timestep}

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None


def make_mujoco_env(camera_names: list[str]) -> MujocoEnv:
    """Build the ``stack_cubes`` ``MujocoSim`` and wrap it for the protocol."""
    sim = MujocoSim(
        package_assets_path('assets/mujoco/franka_table.xml'),
        positronic.cfg.simulator.stack_cubes_loaders(),
        camera_resolution=(64, 64),
        camera_fps=None,
    )
    return MujocoEnv(sim, camera_names)


class StackCubesAdapter(WireCommandAdapter):
    """The reference adapter: canonical Franka commands/observations <-> the MujocoSim raw payloads."""

    def __init__(self, camera_dict: dict[str, str]):
        super().__init__()
        self._camera_dict = camera_dict  # logical observation name -> the env's model camera name

    def _reset_token(self, context: dict[str, Any]) -> Any:
        return context.get('eval.seed')

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        state = MujocoFrankaState()
        ee_pose = geom.Transform3D(raw_obs['ee_pos'], geom.Rotation.from_quat(raw_obs['ee_quat']))
        state.encode(raw_obs['q'], raw_obs['dq'], ee_pose)
        state.array[14 + 7] = float(raw_obs['status'])
        obs: dict[str, Any] = {'robot_state': state, keys.GRIP: float(raw_obs['grip'])}
        for logical, model_name in self._camera_dict.items():
            frame = raw_obs['cameras'][model_name]
            adapter = pimm.shared_memory.NumpySMAdapter(shape=frame.shape, dtype=frame.dtype)
            adapter.array[:] = frame
            obs[logical] = adapter
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {'sim_state': raw_obs['sim_state']}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return None  # native stack_cubes scores downstream — it reports no live terminal


def remote_stack_cubes_eval(host: str, port: int, *, camera_dict: dict[str, str]) -> Eval:
    """Build the remote ``stack_cubes`` eval (embodiment + task) wired to a running env server."""
    # The server is already up (the test fixture owns it), so the proxy just receives its address.
    proxy = RemoteEnvControlSystem(StackCubesAdapter(camera_dict), nullcontext((host, port)))
    embodiment = remote_franka_embodiment(proxy, camera_dict, descriptor='remote.mujoco.franka')
    privileged = {'sim_state': Observation(proxy.privileged['sim_state'], None)}
    task = Task(
        instruction='Pick up the green cube and place it on the red cube.',
        timeout=15.0,
        privileged=privileged,
        reset=proxy.reset,
        done=proxy.done,
    )
    return Eval(embodiment, task)
