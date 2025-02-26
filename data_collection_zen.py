import asyncio
import logging
from typing import Dict, Optional

from cfg import builds, make_config
from hydra_zen import zen, ZenStore

import cfg.env
import cfg.ui
import cfg.hardware.sound

import ironic as ir
from tools.dataset_dumper import DatasetDumper

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def _dataset_dumper(out_dir: str, video_fps: int, metadata: Dict[str, str] = {}):
    # TODO(aluzan): 'relative_mujoco_model_path' to be added to metadata
    return DatasetDumper(out_dir, additional_metadata=metadata, video_fps=video_fps)


def main(ui: ir.ControlSystem,
         env: ir.ControlSystem,
         data_dumper: ir.ControlSystem,
         rerun: bool = False,
         sound: Optional[ir.ControlSystem] = None):

    async def _main():
        ui.bind(
            robot_grip=env.outs.grip,
            robot_position=env.outs.robot_position,
            images=env.outs.frame,
            robot_status=env.outs.robot_status,
        )
        env.bind(
            target_position=ui.outs.robot_target_position,
            target_grip=ui.outs.gripper_target_grasp,
            reset=ui.outs.reset,
        )

        data_dumper.bind(
            image=env.outs.frame,
            target_grip=ui.outs.gripper_target_grasp,
            target_robot_position=ui.outs.robot_target_position,
            start_episode=ui.outs.start_recording,
            end_episode=ui.outs.stop_recording,
            robot_data=env.outs.state,
            env_metadata=env.outs.metadata,
            ui_metadata=ui.outs.metadata,
        )

        components = [ui, env]
        if rerun:
            from tools.rerun_vis import RerunVisualiser
            visualizer = RerunVisualiser()
            visualizer.bind(
                frame=env.outs.frame,
                new_recording=ui.outs.start_recording,
                ext_force_ee=env.outs.ext_force_ee,
                ext_force_base=env.outs.ext_force_base,
                robot_position=env.outs.robot_position,
            )
            components.append(visualizer)

        if sound is not None:
            components.append(
                sound.bind(force=env.outs.ext_force_ee,
                           start_recording=ui.outs.start_recording,
                           stop_recording=ui.outs.stop_recording))

        system = ir.compose(*components)
        await ir.utils.run_gracefully(system)

    asyncio.run(_main())


store = ZenStore()
store(make_config(
    env=cfg.env.umi_env,
    ui=cfg.ui.teleop,
    sound=cfg.hardware.sound.full,
    data_dumper=builds(_dataset_dumper, populate_full_signature=True, video_fps=30),
    rerun=False,
), name="data_collection")


if __name__ == "__main__":
    cfg.env.store.add_to_hydra_store()
    cfg.ui.store.add_to_hydra_store()
    cfg.hardware.sound.store.add_to_hydra_store()
    store.add_to_hydra_store()
    zen(main).hydra_main(config_name="data_collection", config_path=None, version_base="1.2")
