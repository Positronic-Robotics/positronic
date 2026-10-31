import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from positronic_model_server import server_wire, websocket_wire
from positronic_wire import wire
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

import pimm
from pimm.time import RECEIVED_WORLD
from positronic import keys
from positronic.cfg import analysis
from positronic.cli.eval.run import TaskDriver, run_world
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.offboard.server import PolicyServer
from positronic.policy import PolicyEnd
from positronic.policy import keys as policy_keys
from positronic.policy.processors import PauseOnUnavailable
from positronic.policy.remote import RemotePolicy
from positronic.policy.sequential import Sequential
from positronic.vendors.llm.client import Endpoint
from positronic.vendors.llm.motion import Motion
from positronic.vendors.llm.policy import LLMPolicy, Tool
from positronic.vendors.llm.server import LLMModel, deployment


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


def script_move_then_end(monkeypatch, distance: float, ending: str) -> list[dict]:
    """Make the model move by ``distance`` along x and then call ``ending``; return the states it was shown."""
    states = []

    def request(endpoint, runner, model, messages, tools):
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
                    Tool.MOVE_PATH.value,
                    {
                        'waypoints': [
                            {
                                'x': x + distance,
                                'y': y,
                                'z': z,
                                'roll': roll,
                                'pitch': pitch,
                                'yaw': yaw,
                                'gripper': 0.2,
                            }
                        ],
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
    return states


def arm_with_wrist_camera() -> Embodiment:
    robot, camera = Arm(), Camera()
    return Embodiment(
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


@pytest.mark.timeout(30)
@pytest.mark.parametrize('ending,end', [('done', PolicyEnd.COMPLETE), ('give_up', PolicyEnd.GIVE_UP)])
@pytest.mark.parametrize('charge', [False, True])
@pytest.mark.parametrize('distance', [0.01, 0.2])
def test_move_then_end_records_the_policy_end_across_episodes(monkeypatch, tmp_path, ending, end, charge, distance):
    states = script_move_then_end(monkeypatch, distance, ending)
    embodiment = arm_with_wrist_camera()
    policy = Sequential(PauseOnUnavailable(), LLMPolicy(Endpoint('test'), Motion(), camera_keys=(keys.WRIST_IMAGE,)))
    task = Task(instruction_source='Smoke test.', timeout_sec=10, charge_inference_time=charge)
    run_world(embodiment, TaskDriver(lambda: [task, task], policy, LocalDatasetWriter(tmp_path)))
    dataset = LocalDataset(tmp_path)
    assert len(dataset) == 2
    assert len(states) == 4
    assert not (tmp_path / 'policy').exists()
    assert not list(tmp_path.rglob('*.jsonl'))
    for index, episode in enumerate(dataset):
        assert isinstance(episode, Episode)
        assert episode[eval_keys.TERMINATED] is True
        assert episode[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_POLICY
        assert episode[eval_keys.POLICY_END] == end
        assert eval_keys.SUCCESS not in episode
        assert episode[f'{policy_keys.POLICY_META}.{policy_keys.STOP_REASON}'] == ending
        assert episode[f'{policy_keys.POLICY_META}.{policy_keys.HINDSIGHT}'] == 'Small move observed.'
        transcript = episode.static[f'{policy_keys.POLICY_META}.{policy_keys.TRANSCRIPT}']
        assert isinstance(transcript, list)
        assert len([e for e in transcript if e['event'] == 'instructions']) == 1
        requests = [e for e in transcript if e['event'] == 'request']
        responses = [e for e in transcript if e['event'] == 'response']
        before, after = states[index * 2 : index * 2 + 2]
        assert [e['call'] for e in requests] == [1, 2]
        assert [e['call'] for e in responses] == [1, 2]
        assert [e[policy_keys.OBS_TIME_NS] for e in requests] == [
            before[policy_keys.OBS_TIME_NS],
            after[policy_keys.OBS_TIME_NS],
        ]
        assert [e['tools'][0]['name'] for e in responses] == [Tool.MOVE_PATH.value, ending]
        assert all(e['cameras'] == [keys.WRIST_IMAGE] for e in requests)
        assert [e['call'] for e in transcript if e['event'] == 'accepted'] == [1, 2]
        # The run returns at the decision after the move, long before the time limit.
        move_sec = next(e['duration_s'] for e in transcript if e['event'] == 'accepted')
        first, last = episode.bounds(RECEIVED_WORLD)
        assert move_sec <= (last - first) / 1e9 <= move_sec + 0.1
        target_x = before['position_m'][0] + min(distance, Motion().max_translation)
        assert after['position_m'][0] == pytest.approx(target_x, abs=1e-7)
        assert after['previous_target']['x'] == pytest.approx(target_x)
        assert after['gripper'] == pytest.approx(0.2)
        remaining = target_x - after['position_m'][0]
        assert after['remaining_translation_m'] == pytest.approx([remaining, 0, 0], abs=1e-7)
        final_pose = list(episode[keys.EE_POSE].values())[-1]
        assert final_pose[0] == pytest.approx(target_x)
        assert len(episode[keys.EE_POSE]) > 1


@contextmanager
def served(policy: LLMPolicy, sessions_dir: Path) -> Iterator[RemotePolicy]:
    """A remote policy whose server serves ``policy`` on a free localhost port and writes its sessions there."""
    server = PolicyServer(partial(LLMModel, policy, sessions_dir, 60.0), deployment(policy))
    ws = websocket_wire.WebsocketWire(server_wire.ServedHostPort('localhost', 0))
    ready = threading.Event()
    thread = threading.Thread(target=server.serve, args=([ws], ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(10)
        address = ws.served_address
        assert isinstance(address, server_wire.ServedHostPort)
        yield RemotePolicy('websocket', wire.HostPortAddress('localhost', address.port, wire.SESSION_PATH, ''))
    finally:
        server.shutdown()
        thread.join(10)
        assert not thread.is_alive()


@pytest.mark.timeout(60)
@pytest.mark.parametrize('ending,end', [('done', PolicyEnd.COMPLETE), ('give_up', PolicyEnd.GIVE_UP)])
@pytest.mark.parametrize('charge', [False, True])
def test_a_served_policy_moves_and_ends_each_episode_of_a_remote_run(monkeypatch, tmp_path, ending, end, charge):
    states = script_move_then_end(monkeypatch, 0.01, ending)
    task = Task(instruction_source='Smoke test.', timeout_sec=10, charge_inference_time=charge)
    sessions_dir, dataset_dir = tmp_path / 'sessions', tmp_path / 'dataset'
    sessions_dir.mkdir()
    with served(LLMPolicy(Endpoint('test'), Motion(), camera_keys=(keys.WRIST_IMAGE,)), sessions_dir) as policy:
        run_world(arm_with_wrist_camera(), TaskDriver(lambda: [task, task], policy, LocalDatasetWriter(dataset_dir)))
    dataset = LocalDataset(dataset_dir)
    assert len(dataset) == 2
    assert len(states) == 4
    for index, episode in enumerate(dataset):
        assert isinstance(episode, Episode)
        assert episode[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_POLICY
        assert episode[eval_keys.POLICY_END] == end
        assert episode[f'{policy_keys.SERVER_META}.{policy_keys.TYPE}'] == policy_keys.LLM_TYPE
        assert episode[f'{policy_keys.SERVER_META}.{policy_keys.MODEL}'] == 'test'
        assert analysis.model(episode) == 'test'
        session_id = episode[f'{policy_keys.POLICY_META}.{policy_keys.SESSION_ID}']
        record = json.loads((sessions_dir / f'{session_id}.json').read_text())
        assert record[policy_keys.STOP_REASON] == ending
        assert record[policy_keys.HINDSIGHT] == 'Small move observed.'
        assert [e['call'] for e in record[policy_keys.TRANSCRIPT] if e['event'] == 'request'] == [1, 2]
        move_sec = next(e['duration_s'] for e in record[policy_keys.TRANSCRIPT] if 'duration_s' in e)
        before, after = states[index * 2 : index * 2 + 2]
        assert 'previous_target' not in before
        target_x = before['position_m'][0] + 0.01
        assert after['position_m'][0] == pytest.approx(target_x, abs=1e-7)
        assert after['gripper'] == pytest.approx(0.2)
        assert list(episode[keys.EE_POSE].values())[-1][0] == pytest.approx(target_x)
        # The client plays the move at the local cadence, so the episode lasts as long as the local one does.
        first, last = episode.bounds(RECEIVED_WORLD)
        assert move_sec <= (last - first) / 1e9 <= move_sec + 0.1
