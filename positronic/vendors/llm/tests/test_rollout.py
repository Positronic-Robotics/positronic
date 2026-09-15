import json
from pathlib import Path

import numpy as np
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

import pimm
from positronic import keys
from positronic.cli.eval.run import TaskDriver, run_world
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.serializers import Serializers
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.tests.test_golden_pipeline import FakeGripper, FakeRobot
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import LLMPolicy


class Camera(pimm.ControlSystem):
    def __init__(self):
        self.frames = pimm.ControlSystemEmitter(self)

    def run(self, should_stop, clock):
        frame = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.dtype(np.uint8))
        frame.array[:] = 0
        while not should_stop.value:
            self.frames.emit(frame)
            yield pimm.Sleep(0.05)


@pytest.mark.timeout(30)
@pytest.mark.parametrize('ending', ['done', 'give_up'])
@pytest.mark.parametrize('charge', [False, True])
def test_move_then_policy_stop_commits_real_dataset(monkeypatch, tmp_path, ending, charge):
    states = []

    def request(endpoint, messages, tools, transcript, call, obs_time_ns):
        observations = [
            json.loads(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        state = observations[-1]
        states.append(state)
        if call == 1:
            x, y, z = state['position_m']
            roll, pitch, yaw = state['roll_pitch_yaw_rad']
            return ModelResponse([
                ToolCallPart(
                    'move_to',
                    {
                        'x': x + 0.01,
                        'y': y,
                        'z': z,
                        'roll': roll,
                        'pitch': pitch,
                        'yaw': yaw,
                        'gripper': 0.2,
                        'note': 'Small move.',
                    },
                    tool_call_id='move',
                )
            ])
        assert call == 2
        return ModelResponse([
            ToolCallPart(
                ending, {'reason': 'Finish smoke test.', 'hindsight': 'Small move observed.'}, tool_call_id='finish'
            )
        ])

    monkeypatch.setattr(Endpoint, 'request', request)
    robot, gripper, camera = FakeRobot(), FakeGripper(), Camera()
    embodiment = Embodiment(
        descriptor='test arm',
        observations={
            keys.ROBOT_STATE: Observation(robot.state, Serializers.robot_state),
            keys.GRIP: Observation(gripper.grip, None),
            keys.WRIST_IMAGE: Observation(camera.frames, Serializers.camera_images),
        },
        commands={
            keys.ROBOT_COMMAND: Command(robot.commands, Serializers.robot_command),
            keys.TARGET_GRIP: Command(gripper.target_grip, None),
        },
        prepare_handlers={},
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=robot.robot_meta,
        control_systems=(robot, gripper, camera),
        simulated=True,
    )
    policy = (StopOnFault() | ChunkedSchedule()).wrap(
        LLMPolicy(Endpoint('mock'), Motion(), camera_keys=(keys.WRIST_IMAGE,))
    )
    task = Task(instruction_source='Smoke test.', timeout_sec=10, charge_inference_time=charge)
    run_world(embodiment, TaskDriver(lambda: [task], policy, tmp_path))
    dataset = LocalDataset(tmp_path)
    assert len(dataset) == 1
    episode = dataset[0]
    assert isinstance(episode, Episode)
    assert episode[eval_keys.TERMINATED] is True
    assert episode[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_POLICY
    assert eval_keys.SUCCESS not in episode
    assert episode[f'{policy_keys.POLICY_META}.stop_reason'] == ending
    assert episode[f'{policy_keys.POLICY_META}.hindsight'] == 'Small move observed.'
    transcript = Path(episode[f'{policy_keys.POLICY_META}.transcript'])
    assert transcript.is_file()
    assert len(states) == 2
    assert states[0]['position_m'][0] < states[1]['position_m'][0] <= states[0]['position_m'][0] + 0.01
    assert states[1]['gripper'] == pytest.approx(0.2)
    remaining = states[0]['position_m'][0] + 0.01 - states[1]['position_m'][0]
    assert states[1]['remaining_translation_m'] == pytest.approx([remaining, 0, 0], abs=1e-7)
    final_pose = list(episode[keys.EE_POSE].values())[-1]
    assert final_pose[0] == pytest.approx(states[0]['position_m'][0] + 0.01)
    assert len(episode[keys.EE_POSE]) > 1
