import time

import numpy as np

import geom
from positronic.drivers.roboarm.kinova import KinovaController


if __name__ == '__main__':
    arm = KinovaController('192.168.1.10')
    # arm._execute_reference_action('Retract')
    arm.home()
    with arm:
        time.sleep(0.5)
        q_retract = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
        for _ in range(1):
            try:
                # time.sleep(1)
                # arm.set_target_qpos(q_retract)
                # time.sleep(3)

                pos = arm.current_pose
                print(f'Current pose: {pos}')
                print(f'Current joints: {arm.current_qpos}')

                arm.set_target_pose(pos)
                time.sleep(1)

                pos = geom.Transform3D(pos.translation + [0.00, 0.00, 0.1], pos.rotation)
                arm.set_target_pose(pos)
                time.sleep(5)

            except KeyboardInterrupt:
                arm.stop_event.set()
                break

    print('Done')
