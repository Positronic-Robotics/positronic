from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import tqdm

import pimm
import positronic.cfg.dataset
import positronic.cfg.simulator
from positronic import geom, wire
from positronic.dataset import Dataset, Episode, transforms
from positronic.dataset.ds_player_agent import DsPlayerAgent, DsPlayerStartCommand
from positronic.dataset.ds_writer_agent import DsWriterCommand, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform


class Replay(DsPlayerAgent):
    """Adapts `DsPlayerAgent` to be used as a policy control system."""

    def __init__(self, poll_hz: float = 100.0):
        super().__init__(poll_hz)

        self.robot_state = pimm.FakeReceiver(self)
        self.gripper_state = pimm.FakeReceiver(self)
        self.frames = pimm.ReceiverDict(self, fake=True)

    @property
    def robot_commands(self) -> pimm.ControlSystemEmitter:
        return self.outputs['robot_commands']

    @property
    def target_grip(self) -> pimm.ControlSystemEmitter:
        return self.outputs['target_grip']


class RestoreCommand(transforms.KeyFuncEpisodeTransform):
    def __init__(self):
        super().__init__(robot_commands=self._commands_from_episode)

    @staticmethod
    def _commands_from_episode(episode: Episode) -> Any:
        pose = episode['robot_commands.pose']
        return transforms.Elementwise(pose, RestoreCommand.command_from_pose)

    @staticmethod
    def command_from_pose(pose: Sequence[np.ndarray]) -> Sequence[roboarm.command.CommandType]:
        return transforms.LazySequence(
            pose,
            lambda p: roboarm.command.CartesianPosition(
                geom.Transform3D(translation=p[:3], rotation=geom.Rotation.from_quat(p[3:]))
            ),
        )


@cfn.config(
    dataset=positronic.cfg.dataset.local,
    cameras={'image.handcam_left': 'handcam_left_ph', 'image.wrist': 'wrist_cam_ph', 'image.back_view': 'back_view_ph'},
    mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)
def main(
    dataset: Dataset,
    ep_index: int,
    mujoco_model_path: str,
    cameras: dict[str, str],
    loaders: Sequence[MujocoSceneTransform] = (),
    output_dir: str | None = None,
    show_gui: bool = False,
    fps: int = 30,
):
    dataset = transforms.TransformedDataset(dataset, RestoreCommand(), pass_through=True)
    episode = dataset[ep_index]

    sim = MujocoSim(mujoco_model_path, loaders)
    sim.load_state(episode.static)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    mujoco_cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=fps)
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in cameras.items()}
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    gui = DearpyguiUi() if show_gui else None

    replay = Replay()

    writer_cm = LocalDatasetWriter(Path(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, replay, dataset_writer, cameras, robot_arm, gripper, gui, TimeMode.MESSAGE)
        player_cmd = world.pair(replay.command)

        ds_cmd = pimm.NoOpEmitter()
        if ds_agent is not None:
            ds_cmd = world.pair(ds_agent.command)

            # This is extremely hacky, but it works. The problem to connect them directly
            # is that any emitter can be connected only to one receiver.
            def call_stop(_finished: bool):
                ds_cmd.emit(DsWriterCommand.STOP())
                world.request_stop()
                return True

            _ = world.pair(replay.finished, emitter_wrapper=pimm.map(call_stop))

        sim_iter = world.start([sim, mujoco_cameras, robot_arm, gripper, replay, ds_agent], gui)
        ds_cmd.emit(DsWriterCommand.START(episode.static))
        player_cmd.emit(DsPlayerStartCommand(episode))

        p_bar = tqdm.tqdm(total=round(episode.duration_ns / 1e9, 1), unit='s')

        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()


if __name__ == '__main__':
    cfn.cli(main)
