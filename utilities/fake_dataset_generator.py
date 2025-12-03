import random

import configuronic as cfn
import numpy as np
from tqdm import tqdm

import pimm
from positronic import geom
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, Serializers, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.utils import s3 as pos3

# --- Metadata Templates ---

ACT_META = {
    'inference.policy.name': 'act',
    'inference.observation.name': 'eepose_real',
    'inference.action.name': 'absolute_position',
    'inference.observation.lerobot_features': {
        'observation.state': {'shape': (8,), 'names': ['robot_state.ee_pose', 'grip'], 'dtype': 'float32'},
        'observation.images.left': {'shape': (240, 320, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'},
        'observation.images.side': {'shape': (240, 320, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'},
    },
    'inference.action.lerobot_features': {'action': {'shape': (8,), 'names': ['actions'], 'dtype': 'float32'}},
}

GROOT_META = {
    'inference.policy.name': 'groot',
    'inference.observation.name': 'groot_ee_absolute',
    'inference.action.name': 'absolute_position',
    'inference.observation.gr00t_modality': {
        'state': {
            'robot_position_translation': {'start': 0, 'end': 3},
            'robot_position_quaternion': {'start': 3, 'end': 7, 'rotation_type': 'quaternion'},
            'grip': {'start': 7, 'end': 8},
        },
        'video': {
            'exterior_image_1': {'original_key': 'observation.images.exterior'},
            'wrist_image': {'original_key': 'observation.images.wrist'},
        },
    },
    'inference.action.gr00t_modality': {
        'action': {
            'target_robot_position_translation': {'start': 0, 'end': 3},
            'target_robot_position_quaternion': {'start': 3, 'end': 7, 'rotation_type': 'quaternion'},
            'target_grip': {'start': 7, 'end': 8},
        }
    },
}

OPENPI_META = {
    'inference.policy.name': 'openpi',
    'inference.observation.name': 'openpi_positronic',
    'inference.action.name': 'absolute_position',
}

TASKS = [
    'Pick all the towels one by one from transparent tote and place them into the large grey tote.',
    'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.',
    'Pick all the scissors one by one from transparent tote and place them into the large grey tote.',
]

META_MAP = {'act': ACT_META, 'openpi': OPENPI_META, 'groot': GROOT_META}


class FakeGenerator(pimm.ControlSystem):
    def __init__(
        self,
        num_episodes: int,
        fps: int,
        duration: float,
        policy_meta: dict,
        success_rate: float,
        failure_rate: float,
        min_items: int,
        max_items: int,
    ):
        self.num_episodes = num_episodes
        self.fps = fps
        self.duration = duration
        self.policy_meta = policy_meta
        self.success_rate = success_rate
        self.failure_rate = failure_rate
        self.min_items = min_items
        self.max_items = max_items

        # Emitters
        self.command = pimm.ControlSystemEmitter(self)
        self.image_wrist = pimm.ControlSystemEmitter(self)
        self.image_exterior = pimm.ControlSystemEmitter(self)
        self.robot_state = pimm.ControlSystemEmitter(self)
        self.robot_command = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        rate_limiter = pimm.RateLimiter(clock, hz=self.fps)

        for i in tqdm(range(self.num_episodes)):
            if should_stop.value:
                break

            # --- Start Episode ---
            task = random.choice(TASKS)
            total_items = random.randint(self.min_items, self.max_items)

            # Determine outcome
            is_success = random.random() < self.success_rate
            model_failure = random.random() < self.failure_rate

            if is_success and not model_failure:
                successful_items = total_items
                aborted = False
                notes = ''
            else:
                successful_items = random.randint(0, total_items - 1)
                aborted = True
                if model_failure:
                    notes = 'Model failed to grasp object'
                else:
                    notes = "Robot wasn't able to perform the operation."

            static_data = {
                'task': task,
                'eval.total_items': total_items,
                'eval.successful_items': successful_items,
                'eval.aborted': aborted,
                'eval.model_failure': model_failure,
                'eval.notes': notes,
                'eval.tote_placement': random.choice(['left', 'right']),
                'eval.external_camera': random.choices(['left', 'right', 'NA'], [5, 5, 1])[0],
                'inference.policy_fps': self.fps,
                'inference.simulate_timeout': False,
                **self.policy_meta,
            }

            print(f'Starting episode {i + 1}/{self.num_episodes}: {task} (Success: {successful_items}/{total_items})')
            self.command.emit(DsWriterCommand.START(static_data))

            # --- Episode Loop ---
            start_time = clock.now()
            while clock.now() - start_time < self.duration:
                if should_stop.value:
                    break

                t = clock.now()

                # Robot Data
                q = np.sin(t + np.arange(7))  # 7 joints
                # Circular motion for EE pose
                ee_pos = np.array([0.5 + 0.1 * np.cos(t), 0.1 * np.sin(t), 0.3 + 0.05 * np.sin(2 * t)])
                ee_quat = np.array([0, 1, 0, 0])  # Fixed orientation for simplicity
                ee_pose = geom.Transform3D(ee_pos, geom.Rotation.from_quat(ee_quat))

                self.robot_state.emit({'.q': q, '.ee_pose': Serializers.transform_3d(ee_pose)})

                # Robot Command (slightly offset from state)
                cmd_pos = ee_pos + np.array([0.01, 0.01, 0.0])
                cmd_pose = geom.Transform3D(cmd_pos, geom.Rotation.from_quat(ee_quat))
                self.robot_command.emit({'.pose': Serializers.transform_3d(cmd_pose)})

                # Image Data
                for emitter, name in [(self.image_wrist, 'wrist'), (self.image_exterior, 'exterior')]:
                    img = np.zeros((240, 320, 3), dtype=np.uint8)
                    # Moving box
                    cx = int(160 + 50 * np.cos(t + (0 if name == 'wrist' else 1)))
                    cy = int(120 + 50 * np.sin(t + (0 if name == 'wrist' else 1)))

                    # Draw rectangle using numpy slicing
                    x1, x2 = max(0, cx - 20), min(320, cx + 20)
                    y1, y2 = max(0, cy - 20), min(240, cy + 20)
                    img[y1:y2, x1:x2] = (100, 100, 100)

                    # Simple text (just a colored patch to distinguish)
                    if name == 'wrist':
                        img[10:30, 10:30] = (255, 0, 0)  # Red patch for wrist
                    else:
                        img[10:30, 10:30] = (0, 0, 255)  # Blue patch for exterior

                    adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(img, None)
                    emitter.emit(adapter)

                yield pimm.Sleep(rate_limiter.wait_time())

            # --- Stop Episode ---
            self.command.emit(DsWriterCommand.STOP())
            # Give some time for the writer to process the stop command
            yield pimm.Sleep(0.5)


@cfn.config(
    num_episodes=5, fps=15, duration=10.0, policy='groot', success_rate=0.8, failure_rate=0.5, min_items=1, max_items=5
)
def main(
    output_dir: str,
    num_episodes: int,
    fps: int,
    duration: float,
    policy: str,
    success_rate: float,
    failure_rate: float,
    min_items: int,
    max_items: int,
):
    meta = META_MAP[policy]
    print(f'Generating {num_episodes} episodes to {output_dir} mimicking {policy}...')

    writer = LocalDatasetWriter(pos3.upload(output_dir, sync_on_error=True, interval=None))

    with pimm.World() as world:
        agent = DsWriterAgent(writer, time_mode=TimeMode.CLOCK)
        generator = FakeGenerator(num_episodes, fps, duration, meta, success_rate, failure_rate, min_items, max_items)

        # Wire generator to agent
        agent.add_signal('image.wrist', Serializers.camera_images)
        agent.add_signal('image.exterior', Serializers.camera_images)
        agent.add_signal('robot_state')  # Already dict
        agent.add_signal('robot_command')  # Already dict

        world.connect(generator.command, agent.command)
        world.connect(generator.image_wrist, agent.inputs['image.wrist'])
        world.connect(generator.image_exterior, agent.inputs['image.exterior'])
        world.connect(generator.robot_state, agent.inputs['robot_state'])
        world.connect(generator.robot_command, agent.inputs['robot_command'])

        for _ in world.start([agent, generator]):
            pass

    print('Generation complete.')


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli(main)
