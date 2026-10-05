import pytest

import pimm
from pimm.time import RECEIVED_WALL, RECEIVED_WORLD
from positronic.dataset.ds_player_agent import DsPlayerAbortCommand, DsPlayerAgent, DsPlayerStartCommand
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
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

    assert [(ts['playback.scheduled'], value) for ts, value in outputs['a'].emitted] == [(0, 'a1'), (2000, 'a2')]
    assert [(ts['playback.scheduled'], value) for ts, value in outputs['b'].emitted] == [(1000, 'b1')]
    assert [value for _, value in finished.emitted] == [start_cmd]


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

    assert [(ts['playback.scheduled'], value) for ts, value in outputs['a'].emitted] == [(0, 'drop'), (2000, 'keep')]
    assert [(ts['playback.scheduled'], value) for ts, value in outputs['b'].emitted] == [(1000, 'b1')]
    assert finished.emitted, 'Finished command should be emitted when playback completes'


def test_respects_end_timestamp(world):
    outputs = {'a': RecordingEmitter()}
    agent, command_receiver, _ = create_agent(outputs)

    episode = EpisodeContainer(data={'a': DummySignal([1000, 2000, 3000], ['first', 'excluded', 'after'])})
    command_receiver.push(DsPlayerStartCommand(episode, start_ts=1000, end_ts=2000))

    scheduler = world.interleave(agent.run)

    drive_until(scheduler, lambda: len(outputs['a'].emitted) == 1)

    assert [(ts['playback.scheduled'], value) for ts, value in outputs['a'].emitted] == [(0, 'first')]


@pytest.mark.parametrize('timeline', ['world', 'server.wall'])
@pytest.mark.parametrize(
    'start_ts, end_ts, expected',
    [
        (None, None, {'a': [(0, 'a1'), (2000, 'a2'), (4000, 'a3')], 'b': [(1000, 'b1'), (3000, 'b2'), (5000, 'b3')]}),
        (1500, 5000, {'a': [(0, 'a1'), (1500, 'a2')], 'b': [(500, 'b1'), (2500, 'b2')]}),
    ],
)
def test_selected_timeline_controls_bounds_and_scheduling(world, timeline, start_ts, end_ts, expected):
    outputs = {'a': RecordingEmitter(), 'b': RecordingEmitter()}
    agent, command_receiver, finished = create_agent(outputs)
    episode = EpisodeContainer({
        'a': DummySignal([[1000, 1], [3000, 2], [5000, 3]], ['a1', 'a2', 'a3'], timelines=(timeline, 'tick')),
        'b': DummySignal([[2000, 10], [4000, 20], [6000, 30]], ['b1', 'b2', 'b3'], timelines=(timeline, 'tick')),
    })
    command = DsPlayerStartCommand(episode, start_ts=start_ts, end_ts=end_ts, timeline=timeline)
    command_receiver.push(command)
    scheduler = world.interleave(agent.run)
    drive_until(scheduler, lambda: bool(finished.emitted))
    assert {
        name: [(ts['playback.scheduled'], value) for ts, value in output.emitted] for name, output in outputs.items()
    } == expected
    assert [value for _, value in finished.emitted] == [command]


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


@pytest.mark.parametrize('axis', [RECEIVED_WORLD, RECORDED_TIME])
def test_default_axis_plays_new_and_legacy_recordings(world, axis):
    outputs = {'a': RecordingEmitter()}
    agent, commands, finished = create_agent(outputs)
    episode = EpisodeContainer({'a': DummySignal([100, 300], [1, 2], timelines=(axis,))})
    commands.push(DsPlayerStartCommand(episode))
    drive_until(world.interleave(agent.run), lambda: bool(finished.emitted))
    assert [(time['playback.scheduled'], value) for time, value in outputs['a'].emitted] == [(0, 1), (200, 2)]


def test_default_axis_uses_world_receipt(world):
    outputs = {'a': RecordingEmitter()}
    agent, commands, finished = create_agent(outputs)
    episode = EpisodeContainer({
        'a': DummySignal([[100, 1000], [300, 9000]], [1, 2], timelines=(RECEIVED_WORLD, RECEIVED_WALL))
    })
    commands.push(DsPlayerStartCommand(episode))
    drive_until(world.interleave(agent.run), lambda: bool(finished.emitted))
    assert [time['playback.scheduled'] for time, _ in outputs['a'].emitted] == [0, 200]


@pytest.mark.parametrize('axis', [RECEIVED_WALL, 'legacy.clock'])
def test_other_timelines_require_explicit_selection(world, axis):
    outputs = {'a': RecordingEmitter()}
    agent, commands, finished = create_agent(outputs)
    episode = EpisodeContainer({'a': DummySignal([100, 300], [1, 2], timelines=(axis,))})
    commands.push(DsPlayerStartCommand(episode))
    with pytest.raises(ValueError, match='explicit timeline'):
        next(world.interleave(agent.run))
    commands.push(DsPlayerStartCommand(episode, timeline=axis))
    drive_until(world.interleave(agent.run), lambda: bool(finished.emitted))
    assert [time['playback.scheduled'] for time, _ in outputs['a'].emitted] == [0, 200]
