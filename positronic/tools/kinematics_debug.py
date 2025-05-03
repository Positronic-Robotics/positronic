import fire
import mujoco
import numpy as np
import rerun as rr

import ironic as ir
from geom import Transform3D
from positronic.drivers.roboarm.kinova.base import JointCompliantController, KinematicsSolver, wrap_joint_angle


def debug_kinematics(input_filename, rerun):
    recording = rr.dataframe.load_recording(input_filename)
    target_df = recording.view(index='time', contents='/target_position/**').select().read_all().to_pandas()
    pos = []
    for i in range(len(target_df)):
        record = target_df.iloc[i]
        pos.append((record['log_time'].value // 10 ** 6,
                   (record['/target_position/translation:Scalar'],
                    record['/target_position/quat:Scalar'])))

    start_df = recording.view(index='time', contents='/current_joints/**').select().read_all().to_pandas()
    q_start = start_df.iloc[0]['/current_joints:Scalar']

    rr.init('notebook_zero')
    rr.save(rerun)

    torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])
    current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0])
    tau_limit = torque_constant * current_limit_max

    solver, controller = KinematicsSolver(), JointCompliantController(7)
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
    start_time = pos[0][0]

    rr.log('ik/qpos', rr.SeriesPoints(markers="circle", marker_sizes=1))
    rr.log('ik/updates/main', rr.SeriesPoints(markers="cross", marker_sizes=1.0))
    rr.log('ik/updates/null', rr.SeriesPoints(markers="cross", marker_sizes=1.0))

    while data.time < 60 and next_command < len(pos):
        tau = controller.compute_torque(q, dq, tau)
        np.clip(tau, -tau_limit, tau_limit, out=tau)
        data.ctrl[:] = tau_filter.filter(tau)
        mujoco.mj_step(model, data)
        q, dq, tau = data.qpos, data.qvel, data.ctrl
        rr.set_time('sim_time', duration=data.time)

        if start_time + data.time * 10**3 > pos[next_command][0]:
            rr.log('pos/target', rr.Points3D(np.array([p[0] for _, p in pos[:next_command + 1]])))
            # q_ik = solver.inverse(pos[next_command][1], q, max_iters=10000)
            cmd = pos[next_command][1]
            q_ik = solver.inverse_limits(Transform3D(*cmd), q, max_iters=300, clamp=True, debug=True)
            rr.log('ik/qpos', rr.Scalars(q_ik))
            rr.log('pos/cur_target', rr.Points3D(cmd[0], colors=[255, 255, 255]))
            controller.set_target_qpos(q_ik)
            next_command += 1

        if step % 100 == 0:
            rr.log('state/qpos', rr.Scalars(q))
            rr.log('state/qvel', rr.Scalars(dq))
            rr.log('state/tau', rr.Scalars(tau))
            renderer.update_scene(data, camera='viewer')
            rr.log('render', rr.Image(renderer.render()).compress())
        step += 1


main = ir.Config(debug_kinematics)

if __name__ == "__main__":
    fire.Fire(main.override_and_instantiate)
