from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import tqdm

import pimm
import positronic.cfg.dataset
import positronic.cfg.simulator
from positronic import geom
from positronic.dataset import Dataset, Episode, transforms
from positronic.dataset.ds_player_agent import DsPlayerAgent, DsPlayerCommand, DsPlayerStartCommand
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, Serializers, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform


class ReplayController(pimm.ControlSystem):
    def __init__(self, episode: Episode):
        self.episode = episode
        self.player_command = pimm.ControlSystemEmitter[DsPlayerCommand](self)
        self.writer_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.finished = pimm.ControlSystemReceiver(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.player_command.emit(DsPlayerStartCommand(self.episode))
        self.writer_command.emit(DsWriterCommand.START(self.episode.static))
        yield pimm.Pass()
        while not should_stop.value:
            if self.finished.read() is not None:
                self.writer_command.emit(DsWriterCommand.STOP())
                # TODO: This is a hacky way to wait for the writer to finish
                # I think the correct way is to modify the world to wait for all the "pipes" to become empty
                # May be we will need to implement something like "close" on the signals.
                yield pimm.Sleep(0.1)
                return
            else:
                yield pimm.Sleep(1)


class RestoreCommand(transforms.KeyFuncEpisodeTransform):
    def __init__(self):
        super().__init__(robot_commands=self._commands_from_episode)

    @staticmethod
    def _commands_from_episode(episode: Episode) -> Any:
        pose = episode['robot_commands.pose']
        return transforms.Elementwise(pose, RestoreCommand.command_from_pose, names=['robot_commands'])

    @staticmethod
    def command_from_pose(pose: Sequence[np.ndarray]) -> Sequence[roboarm.command.CommandType]:
        return transforms.LazySequence(
            pose,
            lambda p: roboarm.command.CartesianMove(
                geom.Transform3D(translation=p[:3], rotation=geom.Rotation.from_quat(p[3:]))
            ),
        )


@cfn.config(
    dataset=positronic.cfg.dataset.local,
    mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)
def main(
    dataset: Dataset,
    ep_index: int,
    mujoco_model_path: str,
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
    cameras = {
        'handcam_left': MujocoCamera(sim.model, sim.data, 'handcam_left_ph', (320, 240), fps=fps),
        'handcam_right': MujocoCamera(sim.model, sim.data, 'handcam_right_ph', (320, 240), fps=fps),
        'back_view': MujocoCamera(sim.model, sim.data, 'back_view_ph', (320, 240), fps=fps),
        'agent_view': MujocoCamera(sim.model, sim.data, 'agentview', (320, 240), fps=fps),
    }
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    gui = DearpyguiUi() if show_gui else None

    replay = DsPlayerAgent()
    controller = ReplayController(episode)

    writer_cm = LocalDatasetWriter(Path(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        world.connect(replay.outputs['robot_commands'], robot_arm.commands)
        world.connect(replay.outputs['target_grip'], gripper.target_grip)

        ds_agent = None
        if dataset_writer is not None:
            signals_spec = {v: Serializers.image() for v in cameras}
            signals_spec['target_grip'] = None
            signals_spec['robot_commands'] = Serializers.robot_command
            signals_spec['robot_state'] = Serializers.robot_state
            signals_spec['grip'] = None
            ds_agent = DsWriterAgent(dataset_writer, signals_spec, time_mode=TimeMode.MESSAGE)

            for camera_name, camera in cameras.items():
                world.connect(camera.frame, ds_agent.inputs[camera_name])

            world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
            world.connect(gripper.grip, ds_agent.inputs['grip'])
            world.connect(replay.outputs['robot_commands'], ds_agent.inputs['robot_commands'])
            world.connect(replay.outputs['target_grip'], ds_agent.inputs['target_grip'])
            world.connect(controller.writer_command, ds_agent.command)

        if gui is not None:
            for camera_name, camera in cameras.items():
                world.connect(camera.frame, gui.cameras[camera_name])

        world.connect(controller.player_command, replay.command)
        world.connect(replay.finished, controller.finished)

        sim_iter = world.start([sim, *cameras.values(), robot_arm, gripper, replay, ds_agent, controller], gui)
        p_bar = tqdm.tqdm(total=round(episode.duration_ns / 1e9, 1), unit='s')

        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()


if __name__ == '__main__':
    cfn.cli(main)
