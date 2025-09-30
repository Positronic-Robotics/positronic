import time

import configuronic as cfn
import numpy as np
import rerun as rr
from scipy.linalg import orthogonal_procrustes

import pimm
import positronic.cfg.hardware.roboarm
import positronic.cfg.webxr
import positronic.drivers.roboarm.command as roboarm_command
from positronic import geom
from positronic.geom.trajectory import AbsoluteTrajectory, RelativeTrajectory


def _plot_trajectory(trajectory: AbsoluteTrajectory, name: str, color: list[int] | None = None):
    if color is None:
        color = [255, 0, 0, 255]
    points = []

    for idx, pos in enumerate(trajectory):
        rr.set_time('trajectory', sequence=idx)
        points.append(pos.translation)

    rr.log(
        f'trajectory/{name}',
        rr.Points3D(
            positions=np.array(points),
            radii=np.array([0.005]),
            colors=np.array([color]),
        ),
    )


# Arbitrary trajectory for registration
WAYPOINTS = RelativeTrajectory([
    # Initial joint configuration is handled separately
    geom.Transform3D(translation=[0.0, 0.1, 0.1]),
    # YX plane triangle 0.2 side
    geom.Transform3D(translation=[0.0, 0.2, 0.0]),
    geom.Transform3D(translation=[-0.2, 0.0, 0.0]),
    geom.Transform3D(translation=[0.2, -0.2, 0.0]),
    # XY plane square 0.15 side
    geom.Transform3D(translation=[-0.15, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, -0.15, 0.0]),
    geom.Transform3D(translation=[0.15, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, 0.15, 0.0]),
    # XZ plane square 0.1 side
    geom.Transform3D(translation=[-0.1, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, 0.0, -0.1]),
    geom.Transform3D(translation=[0.1, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, 0.0, 0.1]),
    # Not parallel hourglass
    geom.Transform3D(translation=[-0.05, -0.05, -0.05]),
    geom.Transform3D(translation=[0.0, 0.0, 0.05]),
    geom.Transform3D(translation=[-0.05, -0.05, -0.05]),
    geom.Transform3D(translation=[0.0, 0.0, 0.05]),
    geom.Transform3D(translation=[0.1, 0.1, 0.0]),
    # return to start
    geom.Transform3D(translation=[0.0, 0.0, -0.2]),
])


def umi_relative(left_trajectory: AbsoluteTrajectory, right_trajectory: AbsoluteTrajectory):
    """
    Calculate the relative trajectory of the grippers in a frame that doesn't depend on the global reference frame.

    Args:
        left_trajectory: Trajectory of left tracker positions
        right_trajectory: Trajectory of right tracker positions
    Returns:
        AbsoluteTrajectory: Relative transformation trajectory
    """
    assert len(left_trajectory) == len(right_trajectory)
    assert len(left_trajectory) > 0

    # Calculate initial relative gripper transform
    relative_gripper_transform = left_trajectory[0].inv * right_trajectory[0]

    right = []
    left = []
    for i in range(1, len(right_trajectory)):
        # Calculate relative transformation between consecutive right positions
        right_delta = right_trajectory[i - 1].inv * right_trajectory[i]
        left_delta = left_trajectory[i - 1].inv * left_trajectory[i]

        # Apply the relative transformation to the gripper frame
        right_transform = relative_gripper_transform.inv * right_delta * relative_gripper_transform
        left_transform = left_delta
        right.append(right_transform)
        left.append(left_transform)

    return RelativeTrajectory(right)


def _log_tracking_error(robot_trajectory: AbsoluteTrajectory, registered_trajectory: AbsoluteTrajectory):
    robot_pos = np.array([x.translation for x in robot_trajectory])
    registered_pos = np.array([x.translation for x in registered_trajectory])

    error = np.linalg.norm(robot_pos - registered_pos, axis=1)
    print('=' * 100)
    print(f'Max Tracking error: {np.max(error)}')
    print(f'Mean Tracking error: {np.mean(error)}')
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print(f'{percentile}th percentile Tracking error: {np.percentile(error, percentile)}')


def _perform_umi_registration(data):
    """
    Compute the optimal transformation P such that Ai ≈ P^-1 * Bi * P for all i.

    This uses a closed-form solution based on SVD to find the optimal transformation.

    Returns:
        geom.Transform3D: The optimal transformation P
    """
    if not data:
        raise ValueError('No data collected for registration')

    rr.init('registration', spawn=True)

    robot_trajectory = AbsoluteTrajectory([d['robot_position'] for d in data])
    # make it start from the origin
    robot_trajectory = robot_trajectory.to_relative().to_absolute(geom.Transform3D.identity)

    left_trajectory = AbsoluteTrajectory([d['left_gripper'] for d in data])
    right_trajectory = AbsoluteTrajectory([d['right_gripper'] for d in data])

    _plot_trajectory(robot_trajectory, 'target', color=[255, 0, 0, 255])

    umi_trajectory = umi_relative(left_trajectory, right_trajectory).to_absolute(geom.Transform3D.identity)

    translations = np.array([x.translation for x in umi_trajectory])
    translations_target = np.array([x.translation for x in robot_trajectory])

    registration_mtx, _ = orthogonal_procrustes(translations, translations_target)
    registration_rotation = geom.Rotation.from_rotation_matrix(registration_mtx)

    if np.linalg.det(registration_mtx) < 0:
        print('Registration matrix is not a rotation matrix')

    transform = geom.Transform3D(rotation=registration_rotation)

    registered_trajectory = RelativeTrajectory([
        transform.inv * x * transform for x in umi_trajectory.to_relative()
    ]).to_absolute(geom.Transform3D.identity)

    _plot_trajectory(registered_trajectory, 'registered', color=[0, 255, 0, 255])

    print(f'Registration rotation QUAT: {transform.rotation.as_quat}')
    _log_tracking_error(robot_trajectory, registered_trajectory)

    return transform


@cfn.config(webxr=positronic.cfg.webxr.oculus, robot_arm=positronic.cfg.hardware.roboarm.so101)
def perform_registration(webxr, robot_arm):
    """
    This function performs registration procedure.

    Instructions:
    1. Start this function
    2. Connect to WebXR
    3. Install UMI gripper to the robot
    4. Press S button on keyboard and stay near the robot
    """
    current_point = 0
    data = []

    with pimm.World() as w:
        commands = w.pair(robot_arm.commands)
        state = w.pair(robot_arm.state)
        controller_positions = w.pair(webxr.controller_positions)

        w.start([], background=[webxr, robot_arm])

        move_throttler = pimm.utils.RateLimiter(clock=pimm.world.SystemClock(), every_sec=2.0)

        while not w.should_stop:
            if state.read() is None:
                time.sleep(0.001)
                continue
            if controller_positions.read() is None:
                time.sleep(0.001)
                continue
            break

        print('Press Enter after you attach the UMI gripper to the robot...')
        input()

        while not w.should_stop:
            if move_throttler.wait_time() <= 0:
                commands.emit(roboarm_command.CartesianMove(WAYPOINTS[current_point]))
                current_point += 1
                if current_point >= len(WAYPOINTS):
                    break

            controller_pos = controller_positions.value

            assert controller_pos['left'] is not None, 'Left controller position is lost'
            assert controller_pos['right'] is not None, 'Right controller position is lost'

            data.append({
                'left_gripper': controller_pos['left'],
                'right_gripper': controller_pos['right'],
                'robot_position': state.value.ee_pose,
            })

            time.sleep(0.001)

        _perform_umi_registration(data)


if __name__ == '__main__':
    cfn.cli(perform_registration)
