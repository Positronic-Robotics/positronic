import hydra
from omegaconf import DictConfig
import rerun as rr
from tqdm import tqdm

from simulator.mujoco.sim import create_from_config
from inference.state import StateEncoder
from inference.policy import get_policy
from inference.inference import rerun_log_action, rerun_log_observation



@hydra.main(version_base=None, config_path="configs", config_name="sync_policy_runner")
def main(cfg: DictConfig):
    if cfg.rerun:
        rr.init("inference", spawn=False)
        if ':' in cfg.rerun:
            rr.connect(cfg.rerun)
        elif cfg.rerun is not None:
            rr.save(cfg.rerun)

    simulator, renderer, ik = create_from_config(cfg.hardware)

    # Initialize renderer
    renderer.initialize()
    reference_pose = simulator.robot_position
    # Initialize policy and encoders
    policy = get_policy(cfg.inference.checkpoint_path, cfg.get('policy_args', {}))
    policy.to(cfg.inference.device)
    state_encoder = StateEncoder(hydra.utils.instantiate(cfg.inference.state))
    action_decoder = hydra.utils.instantiate(cfg.inference.action)

    steps = cfg.inference.inference_time_sec * cfg.hardware.mujoco.simulation_hz
    frame_count = 0

    for i in tqdm(range(steps)):
        simulator.step()

        # Get observations
        if simulator.ts_sec >= frame_count / cfg.hardware.mujoco.observation_hz:
            frame_count += 1
            rr.set_time_seconds('time', simulator.ts_sec)
            images = renderer.render()

            if cfg.image_name_mapping:
                images = {f"image.{k}": images[v] for k, v in cfg.image_name_mapping.items()}

            # Encode state
            inputs = {
                'robot_position_translation': simulator.robot_position.translation,
                'robot_position_quaternion': simulator.robot_position.quaternion,
                'robot_joints': simulator.joints,
                'ext_force_ee': simulator.ext_force_ee,
                'ext_force_base': simulator.ext_force_base,
                'grip': simulator.grip
            }
            obs = state_encoder.encode(images, inputs)
            for key in obs:
                obs[key] = obs[key].to(cfg.inference.device)
            if len(policy._action_queue) == 0:
                reference_pose = simulator.robot_position
            inputs['robot_position_translation'] = reference_pose.translation
            inputs['robot_position_quaternion'] = reference_pose.quaternion
            # Get policy action
            action = policy.select_action(obs).squeeze(0).cpu().numpy()
            action_dict = action_decoder.decode(action, inputs)

            if len(policy._action_queue) != 0:
                reference_pose = action_dict['target_robot_position']

            if cfg.inference.rerun:
                rerun_log_observation(simulator.ts_sec, obs)
                rerun_log_action(simulator.ts_sec, action)

            # Apply actions
            target_pos = action_dict['target_robot_position']
            joint_positions = ik.recalculate_ik(target_pos)
            if joint_positions is not None:
                simulator.set_actuator_values(joint_positions)

            if 'target_grip' in action_dict:
                simulator.set_grip(action_dict['target_grip'])


    if cfg.rerun:
        rr.disconnect()
    renderer.close()


if __name__ == "__main__":
    main()
