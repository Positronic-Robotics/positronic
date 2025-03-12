import asyncio
import time

import numpy as np

import geom
from positronic.cfg.ui import stub
from positronic.drivers.roboarm.kinova import Kinova
import ironic as ir


async def async_main():
    arm = Kinova('192.168.1.10')
    stub_ui = stub.override(time_len_sec=10.0).instantiate()

    arm.bind(reset=stub_ui.outs.reset, target_position=stub_ui.outs.robot_target_position)
    stub_ui.bind(robot_position=arm.outs.position)
    sys = ir.compose(arm, stub_ui)

    await ir.utils.run_gracefully(sys)


if __name__ == '__main__':
    asyncio.run(async_main())
    print('Done')
