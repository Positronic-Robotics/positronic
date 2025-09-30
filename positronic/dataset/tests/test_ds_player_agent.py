import pytest

from pimm.tests.testing import MockClock
from positronic.dataset.ds_player_agent import DsPlayerAbortCommand, DsPlayerAgent, DsPlayerStartCommand
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.tests.testing_coutils import ManualCommandReceiver, MutableShouldStop, RecordingEmitter, drive_until


def create_agent(outputs: dict[str, RecordingEmitter]) -> tuple[DsPlayerAgent, ManualCommandReceiver, RecordingEmitter]:
    agent = DsPlayerAgent(poll_hz=1e6)
    command_receiver = ManualCommandReceiver()
    agent.command = command_receiver
    agent.outputs.clear()
    agent.outputs.update(outputs)
    finished = RecordingEmitter()
    agent.finished = finished
    return agent, command_receiver, finished


def test_replays_signals_in_time_order():
    outputs = {'a': RecordingEmitter(), 'b': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)

    episode = EpisodeContainer(
        signals={
            'a': DummySignal([1000, 3000], ['a1', 'a2']),
            'b': DummySignal([2000], ['b1']),
        }
    )

    start_cmd = DsPlayerStartCommand(episode, start_ts=1000)
    command_receiver.push(start_cmd)

    clock = MockClock()
    should_stop = MutableShouldStop()
    loop = agent.run(should_stop, clock)

    drive_until(
        loop, clock, lambda: len(outputs['a'].emitted) == 2 and len(outputs['b'].emitted) == 1 and finished.emitted
    )
    should_stop.set(True)
    with pytest.raises(StopIteration):
        next(loop)

    assert outputs['a'].emitted == [(0, 'a1'), (2000, 'a2')]
    assert outputs['b'].emitted == [(1000, 'b1')]
    assert finished.emitted == [(-1, start_cmd)]


def test_start_ts_defaults_to_episode_start():
    outputs = {'a': RecordingEmitter(), 'b': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)

    episode = EpisodeContainer(
        signals={
            'a': DummySignal([1000, 3000], ['drop', 'keep']),
            'b': DummySignal([2000], ['b1']),
        }
    )

    command_receiver.push(DsPlayerStartCommand(episode))

    clock = MockClock()
    should_stop = MutableShouldStop()
    loop = agent.run(should_stop, clock)

    drive_until(
        loop, clock, lambda: len(outputs['a'].emitted) == 2 and len(outputs['b'].emitted) == 1 and finished.emitted
    )
    should_stop.set(True)
    with pytest.raises(StopIteration):
        next(loop)

    assert outputs['a'].emitted == [(0, 'drop'), (2000, 'keep')]
    assert outputs['b'].emitted == [(1000, 'b1')]
    assert finished.emitted, 'Finished command should be emitted when playback completes'


def test_respects_end_timestamp():
    outputs = {'a': RecordingEmitter()}
    agent, command_receiver, _ = create_agent(outputs)

    episode = EpisodeContainer(signals={'a': DummySignal([1000, 2000, 3000], ['first', 'excluded', 'after'])})
    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000, end_ts=2000))

    clock = MockClock()
    loop = agent.run(MutableShouldStop(), clock)

    drive_until(loop, clock, lambda: len(outputs['a'].emitted) == 1)

    assert outputs['a'].emitted == [(0, 'first')]


def test_abort_stops_without_emitting_finished():
    outputs = {'a': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)

    episode = EpisodeContainer(signals={'a': DummySignal([1000, 2000], ['first', 'second'])})
    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000))

    clock = MockClock()
    should_stop = MutableShouldStop()
    loop = agent.run(should_stop, clock)

    drive_until(loop, clock, lambda: len(outputs['a'].emitted) == 1)

    command_receiver.push(DsPlayerAbortCommand())
    clock.advance(next(loop).seconds)

    pending = len(outputs['a'].emitted)
    for _ in range(5):
        clock.advance(next(loop).seconds)
    should_stop.set(True)
    with pytest.raises(StopIteration):
        next(loop)

    assert len(outputs['a'].emitted) == pending
    assert not finished.emitted


def test_raises_for_static_only_output():
    outputs = {'static': RecordingEmitter()}
    agent, command_receiver, _ = create_agent(outputs)

    episode = EpisodeContainer(signals={'dynamic': DummySignal([1000], [1])}, static={'static': 42})

    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000))

    clock = MockClock()
    loop = agent.run(MutableShouldStop(), clock)

    with pytest.raises(ValueError):
        next(loop)
