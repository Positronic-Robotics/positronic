import asyncio
import logging
from typing import Dict, Optional

import fire

import positronic.cfg.env
import positronic.cfg.ui
import positronic.cfg.hardware.sound

import ironic as ir
from positronic.tools.dataset_dumper import DatasetDumper

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


@ir.config
def dataset_dumper(out_dir: str, video_fps: int, metadata: Dict[str, str] = {}, codec: str = 'libx264'):
    return DatasetDumper(out_dir, additional_metadata=metadata, video_fps=video_fps, codec=codec)


async def _main(
        ui: ir.ControlSystem,
        env: ir.ControlSystem,
        data_dumper: ir.ControlSystem,
        rerun: bool = False,
        sound: Optional[ir.ControlSystem] = None
):
    ui.bind(
        robot_grip=env.outs.grip,
        robot_position=env.outs.position,
        images=env.outs.frame,
        robot_status=env.outs.status,
    )
    env.bind(
        target_position=ui.outs.robot_target_position,
        target_grip=ui.outs.gripper_target_grasp,
        reset=ui.outs.reset,
    )

    # TODO: dirty hack for umi support
    def get_right_if_dict(x):
        return x['right'] if isinstance(x, dict) else x

    data_dumper.bind(
        image=env.outs.frame,
        target_grip=ui.outs.gripper_target_grasp,
        target_robot_position=ir.utils.map_port(get_right_if_dict, ui.outs.robot_target_position),
        start_episode=ui.outs.start_recording,
        end_episode=ui.outs.stop_recording,
        robot_data=env.outs.state,
        env_metadata=env.outs.metadata,
        ui_metadata=ui.outs.metadata,
    )

    components = [ui, env]
    if rerun:
        from positronic.tools.rerun_vis import RerunVisualiser
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
            sound.bind(
                force=env.outs.ext_force_ee,
                start_recording=ui.outs.start_recording,
                stop_recording=ui.outs.stop_recording
            )
        )

    system = ir.compose(*components)
    await ir.utils.run_gracefully(system)


main = ir.Config(
    _main,
    env=positronic.cfg.env.umi,
    ui=positronic.cfg.ui.teleop,
    sound=positronic.cfg.hardware.sound.start_stop,
    data_dumper=dataset_dumper.override(video_fps=30, codec='libx264'),
    rerun=False,
)


async def async_main(**kwargs):
    await main.override_and_instantiate(**kwargs)


# TODO: Think through how we can make it a handy standard function and move it to ironic/config.py
def custom_main(**kwargs):
    if 'help' in kwargs:
        del kwargs['help']
        config = main.override(**kwargs)
        print(config)
        return
    asyncio.run(async_main(**kwargs))


if __name__ == "__main__":
    fire.Fire(custom_main)
