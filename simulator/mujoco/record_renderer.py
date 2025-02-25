import os

from tqdm import tqdm
import hydra
import torch
import geom

from simulator.mujoco.sim import create_from_config
from tools.dataset_dumper import SerialDumper


def process_episode(episode_path, cfg, output_dir):
    data = torch.load(episode_path)
    n_frames = len(data['image_timestamp'])

    cfg.hardware.mujoco.model_path = data['mujoco_model_path']
    simulator, renderer, ik = create_from_config(cfg.hardware)

    simulator.reset()
    simulator.load_state(data)

    renderer.initialize()
    dataset_writer = SerialDumper(output_dir, video_fps=cfg.hardware.mujoco.observation_hz)

    dataset_writer.start_episode()

    event_idx = 0
    tqdm_iter = tqdm(total=n_frames, desc=f"Processing {os.path.basename(episode_path)}")
    frames_rendered = 0

    data['image_timestamp'] = data['image_timestamp'] - data['image_timestamp'][0]

    while event_idx < n_frames:
        if data['image_timestamp'][event_idx] <= simulator.ts_ns:
            if cfg.use_ik:
                t = geom.Transform3D(
                    translation=data['target_robot_position_translation'][event_idx],
                    quaternion=data['target_robot_position_quaternion'][event_idx]
                )
                try:
                    t = ik.recalculate_ik(t)
                except Exception as e:
                    print(f"IK failed for {event_idx}")
                    return
                simulator.set_actuator_values(t)
            else:
                simulator.set_actuator_values(data['actuator_values'][event_idx])
            simulator.set_grip(data['target_grip'][event_idx])
            event_idx += 1
            tqdm_iter.update(1)

        tqdm_iter.set_postfix(sim_ts=simulator.ts_sec)
        simulator.step()

        if simulator.ts_sec >= frames_rendered / cfg.hardware.mujoco.observation_hz:
            frames_rendered += 1
            images = renderer.render_frames()

            actual_event_idx = max(0, event_idx - 1)

            dataset_writer.write(
                data = {
                    'actuator_values': simulator.actuator_values,
                    'grip': simulator.grip,
                    'robot_position_translation': simulator.robot_position.translation.copy(),
                    'robot_position_quaternion': simulator.robot_position.quaternion.copy(),
                    'ext_force_ee': simulator.ext_force_ee.copy(),
                    'ext_force_base': simulator.ext_force_base.copy(),
                    'robot_joints': simulator.joints.copy(),
                    'target_grip': data['target_grip'][actual_event_idx].clone(),
                    'target_robot_position_quaternion': data['target_robot_position_quaternion'][actual_event_idx].clone(),
                    'target_robot_position_translation': data['target_robot_position_translation'][actual_event_idx].clone(),
                    'image_timestamp': simulator.ts_ns,
                    'robot_timestamp': simulator.ts_ns,
                    'target_timestamp': simulator.ts_ns,
                },
                video_frames={
                    f'image.{mapped_name}': images[orig_name] for mapped_name, orig_name in cfg.image_name_mapping.items()
                },
            )


    tqdm_iter.close()
    dataset_writer.end_episode()


@hydra.main(version_base=None, config_path="../../configs", config_name="record_renderer")
def main(cfg):
    # Get all episode files from input directory
    input_files = [f for f in os.listdir(cfg.input_dir) if f.endswith('.pt')]
    input_files.sort()  # Process files in order

    print(f"Found {len(input_files)} episodes to process")

    for episode_file in tqdm(input_files, desc="Processing episodes"):
        episode_path = os.path.join(cfg.input_dir, episode_file)
        process_episode(episode_path, cfg, cfg.data_output_dir)


if __name__ == "__main__":
    main()
