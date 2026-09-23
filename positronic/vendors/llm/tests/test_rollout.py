import json

import numpy as np
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

import pimm
from positronic import keys
from positronic.cli.eval.run import TaskDriver, run_world
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.layers import PauseOnUnavailable
from positronic.policy.sequential import Sequential
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import OBS_TIME_NS, LLMPolicy


class Arm(pimm.ControlSystem):
    def __init__(self):
        self.commands = pimm.ControlSystemReceiver(self)
        self.target_grip = pimm.ControlSystemReceiver(self)
        self.state = pimm.ControlSystemEmitter(self)
        self.grip = pimm.ControlSystemEmitter(self)

    def run(self, should_stop, clock):
        position, grip = np.array([0.3, 0, 0.4]), 0.0
        while not should_stop.value:
            command = pimm.value_updated(self.commands)
            if command is not None:
                position = command.pose.translation
            target = pimm.value_updated(self.target_grip)
            if target is not None:
                grip = target
            self.state.emit(make_robot_state(position, np.zeros(7)))
            self.grip.emit(grip)
            yield pimm.Sleep(0.005)


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
@pytest.mark.parametrize('distance', [0.01, 0.2])
def test_move_then_idle_records_until_timeout_across_episodes(monkeypatch, tmp_path, ending, charge, distance):
    states = []

    def request(endpoint, messages, tools):
        observations = [
            json.loads(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        state = observations[-1]
        states.append(state)
        if 'previous_target' not in state:
            x, y, z = state['position_m']
            roll, pitch, yaw = state['roll_pitch_yaw_rad']
            return ModelResponse([
                ToolCallPart(
                    'move_to',
                    {
                        'x': x + distance,
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
        return ModelResponse([
            ToolCallPart(
                ending, {'reason': 'Finish smoke test.', 'hindsight': 'Small move observed.'}, tool_call_id='finish'
            )
        ])

    monkeypatch.setattr(Endpoint, 'request', request)
    robot, camera = Arm(), Camera()
    embodiment = Embodiment(
        descriptor='test arm',
        observations={
            keys.ROBOT_STATE: Observation(robot.state, Serializers.robot_state),
            keys.GRIP: Observation(robot.grip, None),
            keys.WRIST_IMAGE: Observation(camera.frames, Serializers.camera_images),
        },
        commands={
            keys.ROBOT_COMMAND: Command(robot.commands, Serializers.robot_command),
            keys.TARGET_GRIP: Command(robot.target_grip, None),
        },
        prepare_handlers={},
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=None,
        control_systems=(robot, camera),
        simulated=True,
    )
    policy = Sequential(PauseOnUnavailable(), LLMPolicy(Endpoint('test'), Motion(), camera_keys=(keys.WRIST_IMAGE,)))
    task = Task(instruction_source='Smoke test.', timeout_sec=10, charge_inference_time=charge)
    run_world(embodiment, TaskDriver(lambda: [task, task], policy, tmp_path))
    dataset = LocalDataset(tmp_path)
    assert len(dataset) == 2
    assert len(states) == 4
    assert not (tmp_path / 'policy').exists()
    assert not list(tmp_path.rglob('*.jsonl'))
    for index, episode in enumerate(dataset):
        assert isinstance(episode, Episode)
        assert episode[eval_keys.TERMINATED] is False
        assert eval_keys.ENDED_BY not in episode
        assert eval_keys.SUCCESS not in episode
        assert episode.duration_ns / 1e9 == pytest.approx(task.timeout_sec, abs=0.1)
        assert episode[f'{policy_keys.POLICY_META}.stop_reason'] == ending
        assert episode[f'{policy_keys.POLICY_META}.hindsight'] == 'Small move observed.'
        transcript = episode.static[f'{policy_keys.POLICY_META}.transcript']
        assert isinstance(transcript, list)
        assert len([e for e in transcript if e['event'] == 'instructions']) == 1
        requests = [e for e in transcript if e['event'] == 'request']
        responses = [e for e in transcript if e['event'] == 'response']
        before, after = states[index * 2 : index * 2 + 2]
        assert [e['call'] for e in requests] == [1, 2]
        assert [e['call'] for e in responses] == [1, 2]
        assert [e[OBS_TIME_NS] for e in requests] == [before[OBS_TIME_NS], after[OBS_TIME_NS]]
        assert [e['tools'][0]['name'] for e in responses] == ['move_to', ending]
        assert all(e['cameras'] == [keys.WRIST_IMAGE] for e in requests)
        assert [e['call'] for e in transcript if e['event'] == 'accepted'] == [1, 2]
        target_x = before['position_m'][0] + min(distance, Motion().max_translation)
        assert after['position_m'][0] == pytest.approx(target_x, abs=1e-7)
        assert after['previous_target']['x'] == pytest.approx(target_x)
        assert after['gripper'] == pytest.approx(0.2)
        remaining = target_x - after['position_m'][0]
        assert after['remaining_translation_m'] == pytest.approx([remaining, 0, 0], abs=1e-7)
        final_pose = list(episode[keys.EE_POSE].values())[-1]
        assert final_pose[0] == pytest.approx(target_x)
        assert len(episode[keys.EE_POSE]) > 1
