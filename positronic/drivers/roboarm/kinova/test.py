import time

import numpy as np

from positronic.drivers.roboarm.kinova import KinovaController


if __name__ == '__main__':
    arm = KinovaController('192.168.1.10')
    # arm._execute_reference_action('Retract')
    arm.home()
    with arm:
        time.sleep(0.5)
        q_retract = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
        arm.set_target_qpos(q_retract)
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                arm.stop_event.set()
                break

    print('Done')
