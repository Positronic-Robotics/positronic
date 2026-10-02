import pytest

import pimm
from positronic.dataset.ds_player_agent import DsPlayerAbortCommand, DsPlayerAgent, DsPlayerStartCommand
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter, drive_until


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


def create_agent(outputs: dict[str, RecordingEmitter]) -> tuple[DsPlayerAgent, ManualCommandReceiver, RecordingEmitter]:
    agent = DsPlayerAgent(poll_hz=1e6)
    command_receiver = ManualCommandReceiver()
    agent.command = command_receiver
    agent.outputs.clear()
    agent.outputs.update(outputs)
    finished = RecordingEmitter()
    agent.finished = finished
    return agent, command_receiver, finished


def test_replays_signals_in_time_order(world):
    outputs = {'a': RecordingEmitter(), 'b': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)

    episode = EpisodeContainer(data={'a': DummySignal([1000, 3000], ['a1', 'a2']), 'b': DummySignal([2000], ['b1'])})

    start_cmd = DsPlayerStartCommand(episode, start_ts=1000)
    command_receiver.push(start_cmd)

    scheduler = world.interleave(agent.run)

    drive_until(
        scheduler, lambda: len(outputs['a'].emitted) == 2 and len(outputs['b'].emitted) == 1 and finished.emitted
    )
    world.request_stop()
    with pytest.raises(StopIteration):
        next(scheduler)

    assert outputs['a'].emitted == [(0, 'a1'), (2000, 'a2')]
    assert outputs['b'].emitted == [(1000, 'b1')]
    assert finished.emitted == [(-1, start_cmd)]


def test_start_ts_defaults_to_episode_start(world):
    outputs = {'a': RecordingEmitter(), 'b': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)

    episode = EpisodeContainer(
        data={'a': DummySignal([1000, 3000], ['drop', 'keep']), 'b': DummySignal([2000], ['b1'])}
    )

    command_receiver.push(DsPlayerStartCommand(episode))

    scheduler = world.interleave(agent.run)

    drive_until(
        scheduler, lambda: len(outputs['a'].emitted) == 2 and len(outputs['b'].emitted) == 1 and finished.emitted
    )
    world.request_stop()
    with pytest.raises(StopIteration):
        next(scheduler)

    assert outputs['a'].emitted == [(0, 'drop'), (2000, 'keep')]
    assert outputs['b'].emitted == [(1000, 'b1')]
    assert finished.emitted, 'Finished command should be emitted when playback completes'


def test_respects_end_timestamp(world):
    outputs = {'a': RecordingEmitter()}
    agent, command_receiver, _ = create_agent(outputs)

    episode = EpisodeContainer(data={'a': DummySignal([1000, 2000, 3000], ['first', 'excluded', 'after'])})
    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000, end_ts=2000))

    scheduler = world.interleave(agent.run)

    drive_until(scheduler, lambda: len(outputs['a'].emitted) == 1)

    assert outputs['a'].emitted == [(0, 'first')]


def test_abort_stops_without_emitting_finished(world):
    outputs = {'a': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)

    episode = EpisodeContainer(data={'a': DummySignal([1000, 2000], ['first', 'second'])})
    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000))

    scheduler = world.interleave(agent.run)

    drive_until(scheduler, lambda: len(outputs['a'].emitted) == 1)

    command_receiver.push(DsPlayerAbortCommand())
    # Pump enough instants for the agent to consume the abort and settle back to polling.
    for _ in range(5):
        next(scheduler)

    pending = len(outputs['a'].emitted)
    for _ in range(5):
        next(scheduler)
    world.request_stop()
    with pytest.raises(StopIteration):
        next(scheduler)

    assert len(outputs['a'].emitted) == pending
    assert not finished.emitted


def test_raises_for_static_only_output(world):
    outputs = {'static': RecordingEmitter()}
    agent, command_receiver, _ = create_agent(outputs)

    episode = EpisodeContainer(data={'dynamic': DummySignal([1000], [1]), 'static': 42})

    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000))

    scheduler = world.interleave(agent.run)

    with pytest.raises(ValueError):
        next(scheduler)
