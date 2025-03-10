import asyncio
import time

import numpy as np

import geom
from positronic.drivers.roboarm.kinova import Kinova
import ironic as ir


async def async_main():
    arm = Kinova('192.168.1.10')

    target_port = ir.OutputPort('target_position')
    arm.bind(target_position=target_port)

    await arm.setup()

    try:
        await arm.handle_reset(None)
        await asyncio.sleep(4)

        pos = (await arm.position()).data
        print(f'Current pose: {pos}')

        D, PERIOD = 0.01, 10.0
        start = time.monotonic()
        t = 0.0
        while t < 5.0:
            t = time.monotonic() - start
            delta = np.array([np.cos(2 * np.pi * t / PERIOD), np.sin(2 * np.pi * t / PERIOD), 0])
            pos = geom.Transform3D(pos.translation + D * delta, pos.rotation)
            await asyncio.gather(target_port.write(ir.Message(pos)), asyncio.sleep(0.1))

    finally:
        await arm.cleanup()


if __name__ == '__main__':
    asyncio.run(async_main())
    print('Done')
