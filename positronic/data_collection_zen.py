import asyncio
import logging
from typing import Dict, Optional

import fire

import cfg.env
import cfg.ui
import cfg.hardware.sound

import ironic as ir
from tools.dataset_dumper import DatasetDumper

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


@ir.config
def dataset_dumper(out_dir: str, video_fps: int, metadata: Dict[str, str] = {}, codec: str = 'libx264'):
    return DatasetDumper(out_dir, additional_metadata=metadata, video_fps=video_fps, codec=codec)


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
            target_robot_position=ir.utils.map_port(lambda x: x['right'], ui.outs.robot_target_position),
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


main = ir.Config(
    main,
    env=cfg.env.umi,
    ui=cfg.ui.teleop,
    sound=cfg.hardware.sound.start_stop,
    data_dumper=dataset_dumper.override(video_fps=30, codec='libx264'),
    rerun=False,
)


# TODO: Think through how we can make it a handy standard function and move it to ironic/config.py
def custom_main(**kwargs):
    if 'help' in kwargs:
        del kwargs['help']
        config = main.override(**kwargs)
        print(config)
        return
    main.override_and_instantiate(**kwargs)


if __name__ == "__main__":
    fire.Fire(custom_main)
