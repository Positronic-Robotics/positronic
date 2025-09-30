import configuronic as cfn
import mujoco
import numpy as np
import rerun as rr
import tqdm

from positronic.drivers.roboarm.kinova.base import JointCompliantController, KinematicsSolver, wrap_joint_angle
from positronic.geom import Transform3D
from positronic.utils.rerun_compat import log_numeric_series


def random_6dof_on_sphere(radius: float = 0.5) -> tuple[list[float], list[float]]:
    """
    Generate a random 6DOF pose (translation + quaternion) on a sphere surface.

    Args:
        radius: (float) Radius of the sphere to sample from

    Returns:
        np.ndarray: 7-element array [x, y, z, qw, qx, qy, qz] representing pose
    """
    x, y, z = np.random.normal(0, 1, 3)
    norm = np.linalg.norm([x, y, z]) / radius
    x, y, z = x / norm, y / norm, z / norm

    # make it half-sphere
    z = np.abs(z)

    qw, qx, qy, qz = np.random.normal(0, 1, 4)
    norm = np.linalg.norm([qw, qx, qy, qz])
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    return ([x, y, z], [qw, qx, qy, qz])


trajectory = [[i * 5000.0, random_6dof_on_sphere()] for i in range(100)]


def debug_kinematics(urdf_path: str, mujoco_model_path: str, rerun: str, trajectory: list[list[float]]):
    rr.init('debug_kinematics')
    rr.save(rerun)

    taus = []

    # non-zero for a warm start
    q_start = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    solver = KinematicsSolver(mujoco_model_path, site_name='end_effector')
    controller = JointCompliantController(7, path=urdf_path, relative_dynamics_factor=1.0)
    model = solver.model
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera='viewer')

    tau = controller.gravity(q_start)
    data.qpos = wrap_joint_angle(q_start, np.zeros_like(q_start))
    data.ctrl[:] = tau
    mujoco.mj_forward(model, data)

    tau_filter = JointCompliantController.LowPassFilter(0.1, tau)
    q, dq = data.qpos, data.qvel

    step, next_command = 0, 0
    start_time = 0

    rr.log('ik/qpos', rr.SeriesPoints(markers='circle', marker_sizes=1))
    rr.log('ik/updates/main', rr.SeriesPoints(markers='cross', marker_sizes=1.0))
    rr.log('ik/updates/null', rr.SeriesPoints(markers='cross', marker_sizes=1.0))

    p_bar = tqdm.tqdm(total=min(trajectory[-1][0] / 1000, 600))

    while data.time < 600 and next_command < len(trajectory):
        p_bar.update(model.opt.timestep)
        tau = controller.compute_torque(q, dq, tau)
        data.ctrl[:] = tau_filter.filter(tau)
        mujoco.mj_step(model, data)
        q, dq, tau = data.qpos, data.qvel, data.ctrl
        rr.set_time('sim_time', duration=data.time)
        ee_pose = solver.forward(q)
        rr.log('pos/ee', rr.Points3D(ee_pose.translation, colors=[255, 255, 255]))
        taus.append(tau.copy())

        if start_time + data.time * 10**3 > trajectory[next_command][0]:
            rr.log('pos/target', rr.Points3D(np.array([p[0] for _, p in trajectory[: next_command + 1]])))
            cmd = trajectory[next_command][1]
            q_ik = solver.inverse(Transform3D(translation=cmd[0], rotation=cmd[1]), q, max_iters=300)
            log_numeric_series('ik/qpos', q_ik)
            rr.log('pos/cur_target', rr.Points3D(cmd[0], colors=[255, 255, 255]))
            controller.set_target_qpos(q_ik)
            next_command += 1

        if step % 100 == 0:
            log_numeric_series('state/qpos', q)
            log_numeric_series('state/qvel', dq)
            log_numeric_series('state/tau', tau)
            renderer.update_scene(data, camera='viewer')
            rr.log('render', rr.Image(renderer.render()).compress())
        step += 1

    taus = np.array(taus)

    print('Max/min tau:')
    for i, (max_tau, min_tau) in enumerate(zip(np.max(taus, axis=0), np.min(taus, axis=0), strict=False)):
        print(f'Joint {i}: [{min_tau:.2f}, {max_tau:.2f}]')


main = cfn.Config(debug_kinematics, trajectory=trajectory)

if __name__ == '__main__':
    cfn.cli(main)
