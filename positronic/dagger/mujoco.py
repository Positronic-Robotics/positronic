"""The simulated rig of the DAgger rounds on CPU: ``MujocoSim`` in the process, with the ``lift_cube`` scene and its
scripted expert as the teleoperator."""

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from eval_vocabulary.episode import GT_STATE_SIGNAL

import pimm
from positronic import wire
from positronic.cfg.embodiment import mujoco_franka
from positronic.dagger.rounds import Rig
from positronic.eval import Embodiment
from positronic.policy.base import Policy
from positronic.policy.harness import Harness, Rollout
from positronic.simulator.mujoco.expert import LiftCubeExpert
from positronic.simulator.mujoco.lift_cube import CUBE, CUBE_NAME, LiftCubeDone
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform

# The rate of the ground truth: once for each step of the expert.
GT_STATE_FPS = 50.0


class MujocoRig(Rig):
    """``MujocoSim`` of the model ``mujoco_model_path`` with the scene of ``loaders``, on a virtual clock, for each
    Collect step. The embodiment has the cameras ``cameras``: the name of each observation, and the camera of the
    model that gives it. The teleoperator is ``LiftCubeExpert`` with the license ``license``. The helper gets the
    ground truth of the cube, and an episode ends with success when the cube is up."""

    def __init__(
        self,
        mujoco_model_path: str,
        loaders: Sequence[MujocoSceneTransform],
        license: str,
        *,
        cameras: Mapping[str, str] | None = None,
    ):
        self._model_path = mujoco_model_path
        self._loaders = tuple(loaders)
        self._license = license
        self._cameras = dict(cameras or {})

    def _sim(self) -> MujocoSim:
        return MujocoSim(
            self._model_path, self._loaders, objects={CUBE_NAME: CUBE.body_name}, gt_state_fps=GT_STATE_FPS
        )

    @cached_property
    def _unrun(self) -> Embodiment:
        return mujoco_franka(self._sim(), self._cameras)

    @property
    def observations(self) -> Collection[str]:
        return self._unrun.observations.keys()

    @property
    def commands(self) -> Collection[str]:
        return self._unrun.commands.keys()

    @contextmanager
    def harness(self, *, takes_over_at_start: bool) -> Iterator[tuple[Callable[[Rollout], dict[str, Any]], Policy]]:
        sim = self._sim()
        embodiment = mujoco_franka(sim, self._cameras)
        done = LiftCubeDone()
        with pimm.World(virtual_time=True) as world:
            harness = Harness(embodiment)
            world.connect(sim.gt_state, done.gt_state)
            wire.wire_embodiment(world, harness, embodiment, privileged={GT_STATE_SIGNAL: sim.gt_state}, done=done.done)
            caller = world.pair(harness.perform_task)
            loop = world.start([harness, done, *embodiment.control_systems])

            def perform(rollout: Rollout) -> dict[str, Any]:
                answer = caller(rollout)
                while not answer.done():
                    next(loop)
                return answer.result()

            try:
                yield perform, LiftCubeExpert(self._license, takes_over_at_start=takes_over_at_start)
            finally:
                world.request_stop()
                list(loop)
