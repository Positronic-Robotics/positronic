import os
from typing import Dict

import fire
import torch
from tqdm import tqdm

import configuronic as cfgc
import geom
import positronic.cfg.simulator
from positronic.simulator.mujoco.sim import MujocoSimulatorEnv
from positronic.tools.dataset_dumper import SerialDumper


def process_episode(
    episode_path: str,
    output_dir: str,
    env: MujocoSimulatorEnv,
    observation_hz: int,
    use_ik: bool,
    image_name_mapping: Dict[str, str],
):
    data = torch.load(episode_path)
    n_frames = len(data['image_timestamp'])

    env.simulator.reset()
    env.simulator.load_state(data)

    dataset_writer = SerialDumper(output_dir, video_fps=observation_hz)

    dataset_writer.start_episode()

    event_idx = 0
    tqdm_iter = tqdm(total=n_frames, desc=f"Processing {os.path.basename(episode_path)}")
    frames_rendered = 0

    data['image_timestamp'] = data['image_timestamp'] - data['image_timestamp'][0]

    while event_idx < n_frames:
        if data['image_timestamp'][event_idx] <= env.simulator.ts_ns:

            if use_ik:
                target_robot_position = geom.Transform3D(
                    translation=data['target_robot_position_translation'][event_idx],
                    quaternion=data['target_robot_position_quaternion'][event_idx])
                try:
                    actuator_values = env.inverse_kinematics.recalculate_ik(target_robot_position)
                except Exception:
                    print(f"IK failed for {event_idx}")
                    return
            else:
                actuator_values = data['actuator_values'][event_idx]

            env.simulator.set_actuator_values(actuator_values)
            env.simulator.set_grip(data['target_grip'][event_idx])
            event_idx += 1
            tqdm_iter.update(1)

        tqdm_iter.set_postfix(sim_ts=env.simulator.ts_sec)
        env.simulator.step()

        if env.simulator.ts_sec >= frames_rendered / observation_hz:
            frames_rendered += 1
            images = env.renderer.render_frames()

            actual_event_idx = max(0, event_idx - 1)

            target_robot_position_q = data['target_robot_position_quaternion'][actual_event_idx].clone()
            target_robot_position_t = data['target_robot_position_translation'][actual_event_idx].clone()

            dataset_writer.write(
                data={
                    'actuator_values': env.simulator.actuator_values,
                    'grip': env.simulator.grip,
                    'robot_position_translation': env.simulator.robot_position.translation.copy(),
                    'robot_position_rotation': env.simulator.robot_position.rotation.as_quat.copy(),
                    'ext_force_ee': env.simulator.ext_force_ee.copy(),
                    'ext_force_base': env.simulator.ext_force_base.copy(),
                    'robot_joints': env.simulator.joints.copy(),
                    'target_grip': data['target_grip'][actual_event_idx].clone(),
                    'target_robot_position_quaternion': target_robot_position_q,
                    'target_robot_position_translation': target_robot_position_t,
                    'image_timestamp': env.simulator.ts_ns,
                    'robot_timestamp': env.simulator.ts_ns,
                    'target_timestamp': env.simulator.ts_ns,
                },
                video_frames={
                    f'image.{mapped_name}': images[orig_name]
                    for mapped_name, orig_name in image_name_mapping.items()
                },
            )

    tqdm_iter.close()
    dataset_writer.end_episode()


@cfgc.config(observation_hz=60,
             use_ik=False,
             env=positronic.cfg.simulator.simulator,
             image_name_mapping={
                 'front': 'handcam_front',
                 'back': 'handcam_back',
             })
def main(input_dir: str, output_dir: str, env: MujocoSimulatorEnv, observation_hz: int, use_ik: bool,
         image_name_mapping: Dict[str, str]):
    # Get all episode files from input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]

    print(f"Found {len(input_files)} episodes to process")
    env.renderer.initialize()

    for episode_file in tqdm(input_files, desc="Processing episodes"):
        episode_path = os.path.join(input_dir, episode_file)
        process_episode(episode_path, output_dir, env, observation_hz, use_ik, image_name_mapping)


if __name__ == "__main__":
    fire.Fire(main.override_and_instantiate)
