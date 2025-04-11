from typing import Dict
import fire
import rerun as rr
from tqdm import tqdm

from positronic.inference.action import ActionDecoder
from positronic.simulator.mujoco.sim import MujocoSimulatorEnv
from positronic.inference.state import StateEncoder
from positronic.inference.inference import rerun_log_action, rerun_log_observation

import ironic as ir
import positronic.cfg.inference.state
import positronic.cfg.inference.action
import positronic.cfg.inference.policy
import positronic.cfg.simulator


image_mapping = {
    'back': 'handcam_back',
    'front': 'handcam_front',
}


def run_policy_in_simulator(  # noqa: C901  Function is too complex
        env: MujocoSimulatorEnv,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        policy,
        rerun_path: str,
        inference_time_sec: float,
        observation_hz: float,
        image_name_mapping: Dict[str, str],
        device: str,
):
    if rerun_path:
        rr.init("inference", spawn=False)
        rr.save(rerun_path)

    policy = policy.to(device)

    simulator, renderer, ik = env.simulator, env.renderer, env.inverse_kinematics
    # Initialize renderer
    renderer.initialize()
    reference_pose = simulator.robot_position

    steps = int(inference_time_sec * (1 / simulator.model.opt.timestep))
    frame_count = 0

    for i in tqdm(range(steps)):
        simulator.step()

        # Get observations
        if simulator.ts_sec >= frame_count / observation_hz:
            frame_count += 1
            rr.set_time_seconds('time', simulator.ts_sec)
            images = renderer.render()

            if image_name_mapping:
                images = {f"image.{k}": images[v] for k, v in image_name_mapping.items()}

            # Encode state
            inputs = {
                'robot_position_translation': simulator.robot_position.translation,
                'robot_position_rotation': simulator.robot_position.rotation.as_quat,
                'robot_joints': simulator.joints,
                'ext_force_ee': simulator.ext_force_ee,
                'ext_force_base': simulator.ext_force_base,
                'grip': simulator.grip,
                # TODO: following will be gone if we add support for state/action history
                'reference_robot_position_translation': reference_pose.translation,
                'reference_robot_position_quaternion': reference_pose.rotation.as_quat
            }
            obs = state_encoder.encode(images, inputs)
            for key in obs:
                obs[key] = obs[key].to(device)

            # Get policy action
            action = policy.select_action(obs).squeeze(0).cpu().numpy()
            action_dict = action_decoder.decode(action, inputs)

            if rerun_path:
                rerun_log_observation(simulator.ts_ns, obs)
                rerun_log_action(simulator.ts_ns, action)

            # Apply actions
            target_pos = action_dict['target_robot_position']

            # TODO: (aluzan) this is the most definitely will go to inference next PR
            if policy.chunk_start():
                reference_pose = target_pos

            joint_positions = ik.recalculate_ik(target_pos)
            if joint_positions is not None:
                simulator.set_actuator_values(joint_positions)

            if 'target_grip' in action_dict:
                simulator.set_grip(action_dict['target_grip'])

    if rerun_path:
        rr.disconnect()
    renderer.close()


run = ir.Config(
    run_policy_in_simulator,
    env=positronic.cfg.simulator.simulator,
    state_encoder=positronic.cfg.inference.state.end_effector_back_front,
    action_decoder=positronic.cfg.inference.action.relative_robot_position,
    policy=positronic.cfg.inference.policy.act,
    rerun_path="rerun.rrd",
    inference_time_sec=10,
    observation_hz=60,
    image_name_mapping=image_mapping,
    device="cuda",
)


if __name__ == "__main__":
    fire.Fire(run.override_and_instantiate)
