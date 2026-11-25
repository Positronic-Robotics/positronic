"""The pi0.5 bottles server, run against a stand-in for the author's openpi server in the fork's interpreter."""

import bisect
import json
import os
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from positronic_model_server import server_wire, websocket_wire
from positronic_model_server.protocol import AUTH_HEADER, bearer
from positronic_wire import websocket, wire

from positronic import keys
from positronic.offboard import roboarena
from positronic.offboard import server as offboard_server
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.policy import keys as policy_keys
from positronic.policy import spec
from positronic.policy.base import Step
from positronic.policy.executor import Executor
from positronic.policy.tests.fakes import DueAnswer
from positronic.vendors import pi05_abc
from positronic.vendors.pi05_abc import codecs, server
from positronic.vendors.pi05_abc.tests import author_server_stand_in

TOKEN = 'run-token'
LOG = 'backend.jsonl'


def _fork(directory: Path) -> Path:
    """A fork checkout whose interpreter runs the stand-in in place of the author's server."""
    fork = directory / 'fork'
    (fork / server.FORK_OPENPI).mkdir(parents=True)
    python = fork / server.FORK_PYTHON
    python.parent.mkdir(parents=True)
    stand_in = author_server_stand_in.__file__
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stand_in}" "{directory / LOG}" "$@"\n')
    python.chmod(0o755)
    return fork


def _log(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / LOG).read_text().splitlines()]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _infer_in_a_session(policy_server: PolicyServer, obs: dict[str, Any], query: str = '') -> list[dict[str, Any]]:
    """The commands one session on `query` gets for `obs`, from `policy_server` behind the run's token."""
    wires = [websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))]
    ready = threading.Event()
    thread = threading.Thread(target=policy_server.serve, args=(wires, ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=30)
        served = wires[0].served_address
        assert isinstance(served, server_wire.ServedHostPort)
        address = wire.HostPortAddress('127.0.0.1', served.port, wire.SESSION_PATH, query)
        client = InferenceClient(websocket.WebsocketClientWire(), address, headers={AUTH_HEADER: bearer(TOKEN)})
        client.keepalive()
        session = client.new_session()
        try:
            return session.infer(obs)
        finally:
            session.close()
    finally:
        policy_server.shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive()


def _obs(task: str = codecs.TASK) -> dict[str, Any]:
    return {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 1.0,
        keys.EXTERIOR_IMAGE: np.full((300, 400, 3), 40, dtype=np.uint8),
        keys.WRIST_LEFT_IMAGE: np.full((300, 400, 3), 120, dtype=np.uint8),
        keys.WRIST_RIGHT_IMAGE: np.full((300, 400, 3), 200, dtype=np.uint8),
        keys.TASK: task,
    }


def test_a_session_reaches_the_authors_server_in_its_fields_and_gets_the_whole_chunk(tmp_path):
    checkpoint = tmp_path / 'pi05-abc-130k-task-finetune-16000'
    fork = _fork(tmp_path)
    port = _free_port()
    model = server.pi05_abc_model.override(checkpoint=str(checkpoint), fork=str(fork), port=port)

    actions = _infer_in_a_session(PolicyServer(model, server.pipeline, auth_token=TOKEN), _obs(), 'crop=none')

    assert len(actions) == len(author_server_stand_in.CHUNK)
    for action, expected in zip(actions, author_server_stand_in.CHUNK, strict=True):
        np.testing.assert_allclose(action[keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)].positions, expected[:6])
        np.testing.assert_allclose(
            action[keys.arm_channel(keys.ROBOT_COMMAND, keys.RIGHT_ARM)].positions, expected[7:13]
        )
        assert action[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == pytest.approx(1.0 - expected[6])
        assert action[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == pytest.approx(1.0 - expected[13])
    # One warm-up request comes before the session's own.
    started, _, request = _log(tmp_path)
    fields = request[author_server_stand_in.REQUEST]
    # rules-allow: hardcoded-keys — the fields and flags of the author's server as its code spells them, pinned apart
    # from the package constants so a typo in a constant fails here.
    images = fields.pop('images')
    assert fields == {
        'state': [*range(6), 1.0, *range(6, 12), 0.0],
        'prompt': 'put the plastic bottles in the bin',
        roboarena.ENDPOINT: roboarena.INFER,
    }
    # The 4:3 views fill ABC's 224x168 view, channels first, which openpi pads to its square input.
    assert {name: view['shape'] for name, view in images.items()} == dict.fromkeys(
        ('top', 'left', 'right'), [3, 168, 224]
    )
    assert {name: round(view['mean']) for name, view in images.items()} == {'top': 40, 'left': 120, 'right': 200}
    assert started[author_server_stand_in.CWD] == str(fork / 'third_party/policy/openpi')
    script, *flags = started[author_server_stand_in.ARGV]
    assert script == str(fork / 'yam_abc_reproduce/deploy/servers/openpi_server.py')
    assert dict(zip(flags[::2], flags[1::2], strict=True)) == {
        '--host': '127.0.0.1',
        '--port': str(port),
        '--config': 'pi05_abc130k',
        '--checkpoint': str(checkpoint),
        '--norm-stats-path': str(checkpoint / 'assets/put_the_plastic_bottles_in_the_bin/norm_stats.json'),
        '--prompt': 'put the plastic bottles in the bin',
        '--image-key-map': 'top=image,left=left_wrist,right=right_wrist',
        '--flatten-prefix': 'observation/',
        '--state-key': 'observation/state',
    }
    assert not _alive(started[author_server_stand_in.PID]), 'the server left the backend running after it stopped'


def test_a_task_other_than_the_bottles_is_refused():
    codec = server.pipeline().codec
    with pytest.raises(ValueError, match='has no prompt for the task'):
        codec.encode(_obs('sweep away paper scraps from the table'))


# rules-allow: hardcoded-keys — the task instructions of ABC's sim and the task strings of ABC's real data, pinned apart
# from the package constants so a typo in a constant fails here.
ABC4_INSTRUCTION_PROMPTS = [
    ('put the plastic bottles in the bin', 'put the plastic bottles in the bin'),
    ('sweep away paper scraps from the table', 'sweep away the paper scraps from the table'),
    ('turn mug right side up', 'turn the mug right side up'),
    ('load plates into tabletop dish rack', 'load the plates into the dish rack'),
]
# rules-allow: hardcoded-keys — pinned apart from the package constants, as above.
ABC11_INSTRUCTION_PROMPTS = [
    *ABC4_INSTRUCTION_PROMPTS,
    ('unload the plates from the dish rack', 'unload the plates from the dish rack'),
    ('load the cups into the dish rack', 'load the cups into the dish rack'),
    ('put the screwdriver in the bin', 'put the screwdriver in the bin'),
    ('place and organize the fake fruits in the fruit bowl', 'place and organize the fake fruits in the fruit bowl'),
    ('build the wood block tower', 'build the wood block tower'),
    ('place the snacks into the paper bag', 'place the snacks into the paper bag'),
    ('take the snacks out of the paper bag', 'take the snacks out of the paper bag'),
]
# rules-allow: hardcoded-keys — pinned apart from the package constants, as above.
ABC15_INSTRUCTION_PROMPTS = [
    *ABC11_INSTRUCTION_PROMPTS,
    ('unload the cups from the dish rack', 'unload the cups from the dish rack'),
    ('set the dinner table for one person', 'set the dinner table for one person'),
    ('turn the large container upside down', 'turn the large container upside down'),
    ('unscrew the bottle caps', 'unscrew the bottle caps'),
]


@dataclass(frozen=True)
class MultiTaskCommand:
    """A serve command of a checkpoint trained on several tasks, and what it must send and load."""

    pipeline: Any
    model: Any
    prompts: list[tuple[str, str]]
    norm_stats: str


def _multi_task_command(name: str, prompts: list[tuple[str, str]], norm_stats: str) -> MultiTaskCommand:
    command = server.COMMANDS[name]
    return MultiTaskCommand(command.kwargs['pipeline'], command.kwargs['model'], prompts, norm_stats)


MULTI_TASK_COMMANDS = {
    'abc4_30k': _multi_task_command('abc4_30k', ABC4_INSTRUCTION_PROMPTS, 'assets/abc4_yam/norm_stats.json'),
    'abc11_30k': _multi_task_command('abc11_30k', ABC11_INSTRUCTION_PROMPTS, 'assets/abc11_yam/norm_stats.json'),
    'abc15_30k': _multi_task_command('abc15_30k', ABC15_INSTRUCTION_PROMPTS, 'assets/abc15_yam/norm_stats.json'),
    'abc15_last_command': _multi_task_command(
        'abc15_last_command', ABC15_INSTRUCTION_PROMPTS, 'assets/abc15_yam/norm_stats.json'
    ),
}
COMMAND_PROMPTS = [
    (name, instruction, prompt)
    for name, command in MULTI_TASK_COMMANDS.items()
    for instruction, prompt in command.prompts
]


@pytest.mark.parametrize(('name', 'instruction', 'prompt'), COMMAND_PROMPTS)
def test_a_multi_task_command_sends_the_real_data_prompt_of_each_task(name, instruction, prompt):
    codec = MULTI_TASK_COMMANDS[name].pipeline().codec
    assert codec.encode(_obs(instruction))[pi05_abc.PROMPT] == prompt


@pytest.mark.parametrize('name', list(MULTI_TASK_COMMANDS))
def test_a_multi_task_command_refuses_a_task_it_did_not_train_on(name):
    command = MULTI_TASK_COMMANDS[name]
    codec = command.pipeline().codec
    assert codec.meta[codecs.PROMPTS_META] == dict(command.prompts)
    with pytest.raises(ValueError, match='has no prompt for the task'):
        codec.encode(_obs('hang mug on mug rack'))


@pytest.mark.parametrize('name', list(MULTI_TASK_COMMANDS))
def test_a_multi_task_command_loads_its_own_norm_stats(name, tmp_path):
    command = MULTI_TASK_COMMANDS[name]
    command.model(checkpoint=str(tmp_path / 'checkpoint'), fork=str(_fork(tmp_path)), port=_free_port()).close()

    argv = _log(tmp_path)[0][author_server_stand_in.ARGV]
    assert argv[argv.index('--norm-stats-path') + 1] == str(tmp_path / 'checkpoint' / command.norm_stats)


@pytest.mark.parametrize('crop', list(codecs.Crop))
def test_a_session_crop_names_the_boxes_of_that_set(crop):
    meta = server.pipeline(crop=crop.value).codec.meta
    crops = codecs.CROPS[crop]
    assert meta[codecs.CROPS_META] == {key: [b.left, b.top, b.right, b.bottom] for key, b in crops.items()}


def test_an_unknown_crop_is_refused():
    with pytest.raises(ValueError, match='Unknown crop'):
        server.pipeline(crop='center')


def test_the_default_crop_keeps_the_whole_exterior_frame():
    assert server.pipeline().codec.meta[codecs.CROPS_META][keys.EXTERIOR_IMAGE] == [0.0, 0.0, 1.0, 1.0]


def test_a_load_with_relative_paths_reaches_the_server_with_absolute_ones(tmp_path, monkeypatch):
    _fork(tmp_path)
    monkeypatch.chdir(tmp_path)

    server.pi05_abc_model(checkpoint='checkpoint', fork='fork', port=_free_port()).close()

    argv = _log(tmp_path)[0][author_server_stand_in.ARGV]
    assert argv[0] == str(tmp_path / 'fork/yam_abc_reproduce/deploy/servers/openpi_server.py')
    assert argv[argv.index('--checkpoint') + 1] == str(tmp_path / 'checkpoint')
    assert argv[argv.index('--norm-stats-path') + 1].startswith(str(tmp_path / 'checkpoint'))


def test_a_port_another_server_listens_on_fails_the_load_before_the_backend_starts(tmp_path):
    fork = _fork(tmp_path)
    with socket.create_server(('127.0.0.1', 0)) as occupant:
        with pytest.raises(OSError, match='Address already in use'):
            server.pi05_abc_model(
                checkpoint=str(tmp_path / 'checkpoint'), fork=str(fork), port=occupant.getsockname()[1]
            )
    assert not (tmp_path / LOG).exists()


def test_the_default_backend_port_is_free_for_the_standard_wires():
    backend_port = server.pi05_abc_model.kwargs['port']
    wires = (offboard_server.websocket, offboard_server.grpc)
    assert backend_port not in {served.kwargs['served_address'].kwargs['port'] for served in wires}


# One inference of the rig round trip: 72 ms in the model and about 125 ms on the network.
ROUND_TRIP_NS = 197_000_000
PERIOD_NS = round(1e9 / codecs.FPS)
MOTOR = 'motor'


class _Answer(DueAnswer):
    """A `DueAnswer` that knows whether its result was read."""

    read = False

    def result(self):
        self.read = True
        return super().result()


class _Model:
    """Answers each call with a 50-action chunk after `latency_ns`, and records when each call read its observation.

    Action `i` of call `c` is `(c, i)`."""

    def __init__(self, runtime: Executor, latency_ns: int):
        self._runtime = runtime
        self._latency_ns = latency_ns
        self._answers: list[_Answer] = []
        self.observed_ns: list[int] = []
        self.plans: list[Any] = []

    def submit(self, function, obs):
        call = len(self.observed_ns)
        self.observed_ns.append(self._runtime.time_ns)
        self.plans.append(obs.get(policy_keys.RTC.ACTION_PREFIX))
        chunk = [{MOTOR: (call, i)} for i in range(len(author_server_stand_in.CHUNK))]
        self._answers.append(_Answer(self._runtime, chunk, self._runtime.time_ns + self._latency_ns))
        return self._answers[-1]

    def has_ready_answers(self) -> bool:
        return any(answer.done() and not answer.read for answer in self._answers)


def _play(
    latency_ns: int, duration_ns: int = 3_000_000_000, *, last_command: bool = False
) -> tuple[list[tuple[int, Any]], _Model]:
    """The command that the arm holds after each millisecond, while the rig runs the stack that a session declares.

    The stack goes through its wire description, as on the rig. As the harness does, the clock steps by 1 ms, a ready
    answer calls the stack again at the same time, and of two commands at one time the arm keeps the second."""
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    model = _Model(runtime, latency_ns)
    runtime.submit = model.submit
    runtime.has_ready_answers = model.has_ready_answers
    local = server.pipeline(crop='none', last_command=last_command).local
    run = runtime.start(spec.from_spec(local.to_spec()), lambda obs: [])
    emitted: dict[int, Any] = {}
    try:
        for now[0] in range(0, duration_ns, 1_000_000):
            step = run.send({})
            while model.has_ready_answers():
                step = Step({**step.commands, **run.send({}).commands}, step.resume_at_ns)
            if MOTOR in step.commands:
                emitted[now[0]] = step.commands[MOTOR]
    finally:
        run.close()
        runtime.close()
    return sorted(emitted.items()), model


def test_the_arm_never_waits_for_the_next_chunk_on_a_rig_round_trip():
    emitted, model = _play(ROUND_TRIP_NS)

    gaps = np.diff([time for time, _ in emitted])
    assert gaps.max() <= PERIOD_NS + 1_000_000
    # One request for each 15 actions, as the author's evaluation asks.
    assert set(np.diff(model.observed_ns[1:]).tolist()) == {round(server.ACTIONS_PER_REQUEST / codecs.FPS * 1e9)}


def test_each_action_of_a_later_chunk_plays_at_its_time_after_the_observation():
    emitted, model = _play(ROUND_TRIP_NS)

    later = [(time, call, index) for time, (call, index) in emitted if call > 0]
    assert later
    for time, call, index in later:
        assert 0 <= time - (model.observed_ns[call] + index * PERIOD_NS) < PERIOD_NS
    # The actions that the round trip made stale are skipped: a new chunk starts at the action due when it arrives.
    firsts = {call: index for _, call, index in reversed(later)}
    assert set(firsts.values()) == {ROUND_TRIP_NS // PERIOD_NS}


def test_an_instant_answer_plays_the_first_actions_of_each_chunk_as_the_authors_evaluation():
    emitted, _ = _play(latency_ns=0)

    played = [command for _, command in emitted]
    chunks = len(played) // server.ACTIONS_PER_REQUEST
    assert played[: chunks * server.ACTIONS_PER_REQUEST] == [
        (call, index) for call in range(chunks) for index in range(server.ACTIONS_PER_REQUEST)
    ]


def test_a_last_command_session_sends_the_state_as_the_last_command_before_the_first_chunk(tmp_path):
    command = server.COMMANDS['abc15_last_command'].kwargs
    checkpoint, port = tmp_path / 'checkpoint', _free_port()
    model = command['model'].override(checkpoint=str(checkpoint), fork=str(_fork(tmp_path)), port=port)

    _infer_in_a_session(PolicyServer(model, command['pipeline'], auth_token=TOKEN), _obs(), 'crop=none')

    started, warm, request = _log(tmp_path)
    # rules-allow: hardcoded-keys — the field and the flags of the author's server as its code spells them.
    state = request[author_server_stand_in.REQUEST]['state']
    assert request[author_server_stand_in.REQUEST]['last_command'] == state
    assert warm[author_server_stand_in.REQUEST]['last_command'] == [0.0] * codecs.BIMANUAL_SIZE
    flags = started[author_server_stand_in.ARGV][1:]
    options = dict(zip(flags[::2], flags[1::2], strict=True))
    assert options['--config'] == 'pi05_abc130k_last_command'
    assert options['--last-command-key'] == 'observation/last_command'


def test_a_command_without_the_last_command_does_not_send_it(tmp_path):
    port = _free_port()
    model = server.pi05_abc_model.override(checkpoint=str(tmp_path / 'c'), fork=str(_fork(tmp_path)), port=port)

    _infer_in_a_session(PolicyServer(model, server.pipeline, auth_token=TOKEN), _obs(), 'crop=none')

    started, warm, request = _log(tmp_path)
    assert pi05_abc.LAST_COMMAND not in request[author_server_stand_in.REQUEST]
    assert pi05_abc.LAST_COMMAND not in warm[author_server_stand_in.REQUEST]
    assert '--last-command-key' not in started[author_server_stand_in.ARGV]


def test_the_plan_of_each_later_request_starts_with_the_command_that_the_arm_holds_at_its_observation():
    emitted, model = _play(ROUND_TRIP_NS, last_command=True)

    times = [time for time, _ in emitted]
    later = list(zip(model.observed_ns, model.plans, strict=True))[1:]
    assert later
    for observed_ns, plan in later:
        held = emitted[bisect.bisect_right(times, observed_ns) - 1][1]
        assert plan[policy_keys.RTC.PREFIX_ACTIONS][0] == {MOTOR: held}


def test_without_the_last_command_the_client_sends_no_plan():
    _, model = _play(ROUND_TRIP_NS)

    assert model.plans == [None] * len(model.plans)
