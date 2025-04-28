import time
from typing import Dict
import fire
import rerun as rr
from tqdm import tqdm
import numpy as np

import positronic.cfg.hardware
import positronic.cfg.hardware.camera
from positronic.drivers.camera.linuxpy_video import LinuxPyCamera
from positronic.drivers.gripper.dh import DHGripper
from positronic.drivers.roboarm.kinova import KinovaSync
from positronic.inference.action import ActionDecoder
from positronic.inference.state import StateEncoder
from positronic.inference.inference import rerun_log_action, rerun_log_observation

import ironic as ir
import positronic.cfg.inference.state
import positronic.cfg.inference.action
import positronic.cfg.inference.policy
import positronic.cfg.simulator


def get_state(
        env: KinovaSync,
        gripper: DHGripper,
        left_camera: LinuxPyCamera,
        right_camera: LinuxPyCamera,
        reference_pose,
):
    images = {
        'left.image': left_camera.get_frame()['image'],
        'right.image': right_camera.get_frame()['image'],
    }

    inputs = {
        'robot_position_translation': env.get_position().translation,
        'robot_position_rotation': env.get_position().rotation.as_quat,
        'robot_joints': env.get_joint_positions(),
        'grip': gripper.get_grip(),
        # TODO: following will be gone if we add support for state/action history
        'reference_robot_position_translation': reference_pose.translation,
        'reference_robot_position_quaternion': reference_pose.rotation.as_quat
    }

    return images, inputs

def run_policy_in_simulator(  # noqa: C901  Function is too complex
        env: KinovaSync,
        gripper: DHGripper,
        left_camera: LinuxPyCamera,
        right_camera: LinuxPyCamera,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        policy,
        rerun_path: str,
        device: str,
        task: str | None,
):
    if rerun_path:
        rr.init("inference", spawn=False)
        rr.save(rerun_path)

    policy = policy.to(device)
    gripper.sync_setup()
    left_camera.sync_setup()
    right_camera.sync_setup()
    env.setup()


    while True:
        cmd = input("Enter a command: ")
        if cmd == "q":
            break
        elif cmd == "reset":
            env.reset_position()
            env.wait_finish()
            reference_pose = env.get_position()
        elif cmd == "stat":
            print("joints: ", env.get_joint_positions())
            print("position: ", env.get_position())
        elif cmd.startswith("joints"):
            joints = cmd.split(' ')[1:]
            cmd = list(map(float, joints))
            env.execute_joint_command(np.array(cmd))
            reference_pose = env.get_position()

        elif cmd.startswith("e"):
            n_steps = int(cmd.split(' ')[1])
            for _ in range(n_steps):
                current_time = time.monotonic()

                # Get observations
                rr.set_time_seconds('time', current_time)
                images, inputs = get_state(env, gripper, left_camera, right_camera, reference_pose)
                obs = state_encoder.encode(images, inputs)
                for key in obs:
                    obs[key] = obs[key].to(device)

                if task is not None:
                    obs['task'] = task
                # print(inputs)
                # Get policy action
                action = policy.select_action(obs).squeeze(0).cpu().numpy()
                action_dict = action_decoder.decode(action, inputs)

                if rerun_path:
                    rerun_log_observation(current_time, obs)
                    rerun_log_action(current_time, action)

                # Apply actions
                target_pos = action_dict['target_robot_position']

                # TODO: (aluzan) this is the most definitely will go to inference next PR
                target_grip = 1.0 if action_dict['target_grip'] > 0.5 else 0.0

                # print(action_dict['target_robot_position'])
                current_joints = env.get_joint_positions()
                joints = env.solver.inverse(action_dict['target_robot_position'], current_joints)
                # joints[3] = max(joints[3], 3.7)
                for i, joint in enumerate(joints):
                    rr.log(f"target_joints/{i}", rr.Scalar(joint))
                for i, joint in enumerate(current_joints):
                    rr.log(f"current_joints/{i}", rr.Scalar(joint))

                for i, tr in enumerate(action_dict['target_robot_position'].translation):
                    rr.log(f"target_position/translation/{i}", rr.Scalar(tr))
                for i, rot in enumerate(action_dict['target_robot_position'].rotation.as_quat):
                    rr.log(f"target_position/quat/{i}", rr.Scalar(rot))

                env.execute_joint_command(joints)
                gripper.set_grip(target_grip)

                if policy.chunk_start():
                    # env.wait_finish()
                    # reference_pose = env.get_position()
                    reference_pose = target_pos
    if rerun_path:
        rr.disconnect()
    gripper.sync_cleanup()
    env.cleanup()


kinova_sync = ir.Config(
    KinovaSync,
    ip="192.168.1.10",
    relative_dynamics_factor=0.2,
)

gripper = ir.Config(
    DHGripper,
    port="/dev/ttyUSB0",
)


run = ir.Config(
    run_policy_in_simulator,
    env=kinova_sync,
    gripper=gripper,
    left_camera=positronic.cfg.hardware.camera.arducam_left,
    right_camera=positronic.cfg.hardware.camera.arducam_right,
    state_encoder=positronic.cfg.inference.state.end_effector,
    action_decoder=positronic.cfg.inference.action.umi_relative,
    policy=positronic.cfg.inference.policy.act,
    rerun_path="rerun.rrd",
    device="cuda",
    task=None,
)


if __name__ == "__main__":
    fire.Fire(run.override_and_instantiate)
