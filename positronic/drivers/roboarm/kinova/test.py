import asyncio
import time

import numpy as np

import geom
from positronic.drivers.roboarm.kinova import KinovaController
import ironic as ir


async def async_main():
    arm = KinovaController('192.168.1.10')

    target_port = ir.OutputPort('target_position')
    arm.bind(target_position=target_port)

    await arm.setup()

    try:
        await arm.handle_reset(None)
        await asyncio.sleep(5)

        pos = (await arm.position()).data
        print(f'Current pose: {pos}')

        start = time.monotonic()
        D = 0.01
        PERIOD = 10.0
        for i in range(50):
            t = time.monotonic() - start
            delta = np.array([np.cos(2 * np.pi * t / PERIOD), np.sin(2 * np.pi * t / PERIOD), 0.0])
            pos = geom.Transform3D(pos.translation + D * delta, pos.rotation)
            await asyncio.gather(target_port.write(ir.Message(pos)), asyncio.sleep(0.1))

    finally:
        await arm.cleanup()


if __name__ == '__main__':
    asyncio.run(async_main())
    print('Done')
