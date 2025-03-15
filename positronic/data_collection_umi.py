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
        sound: Optional[ir.ControlSystem] = None
):
    ui.bind(
        images=env.outs.frame,
    )
    env.bind(
        target_position=ui.outs.controller_positions,
    )

    data_dumper.bind(
        image=env.outs.frame,
        # TODO: make dataset dumper more generic
        target_robot_position=ir.utils.map_port(lambda x: x['right'], ui.outs.controller_positions),
        target_grip=ir.utils.map_port(lambda _: 0, ui.outs.controller_positions),
        start_episode=ui.outs.start_recording,
        end_episode=ui.outs.stop_recording,
        robot_data=env.outs.state,
        env_metadata=env.outs.metadata,
        ui_metadata=ui.outs.metadata,
    )

    components = [ui, env]

    if sound is not None:
        components.append(
            sound.bind(
                start_recording=ui.outs.start_recording,
                stop_recording=ui.outs.stop_recording
            )
        )

    system = ir.compose(*components)
    await ir.utils.run_gracefully(system)


main = ir.Config(
    _main,
    env=positronic.cfg.env.umi,
    ui=positronic.cfg.ui.teleop_umi,
    sound=positronic.cfg.hardware.sound.start_stop,
    data_dumper=dataset_dumper.override(video_fps=30, codec='libx264'),
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
