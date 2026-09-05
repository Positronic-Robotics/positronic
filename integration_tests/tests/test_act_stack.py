import mujoco as mj
import numpy as np
import pytest

from integration_tests import act_stack
from integration_tests.act_stack import (
    CUBE_POSES,
    EPISODE_SECONDS,
    FINGER_BODIES,
    RECORDED_SIGNALS,
    REFERENCE_FILENAME,
    SUPPORTED,
    TIME_SUFFIX,
    capture,
    check_episode,
    check_stacking,
    checkpoint_url,
    compare_trace,
    is_supported_stack,
    write_npz,
)
from positronic import keys
from positronic.cfg.simulator import STACK_GREEN_CUBE, STACK_RED_CUBE
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter


@pytest.fixture
def recorded_signals(monkeypatch, tmp_path):
    times = np.arange(0, EPISODE_SECONDS * 1_000_000_000 + 1, 100_000_000)
    with DiskEpisodeWriter(tmp_path / 'episode') as writer:
        for name in RECORDED_SIGNALS:
            sample_times = times[::10] if name in (keys.TARGET_EE_POSE, keys.TARGET_GRIP) else times
            for timestamp in sample_times:
                writer.append(name, 0.0, int(timestamp))
    signals = DiskEpisode(tmp_path / 'episode').signals
    monkeypatch.setattr(
        act_stack,
        'cube_trace',
        lambda episode: {
            CUBE_POSES: np.zeros((len(times), 2, 7)),
            CUBE_POSES + TIME_SUFFIX: times - episode.start_ts,
            SUPPORTED: np.ones(len(times), dtype=bool),
            SUPPORTED + TIME_SUFFIX: times - episode.start_ts,
        },
    )
    monkeypatch.setattr(act_stack, 'read_episode', lambda output, seed: EpisodeContainer(signals))
    return signals


@pytest.mark.parametrize('name', [keys.EE_POSE, keys.JOINTS, keys.GRIP])
@pytest.mark.parametrize(
    'retained',
    [
        pytest.param([0], id='only-first'),
        pytest.param(slice(1, None), id='missing-first'),
        pytest.param(slice(None, -1), id='missing-last'),
        pytest.param(np.delete(np.arange(151), 75), id='missing-middle'),
    ],
)
def test_incomplete_observations_fail_success_check_and_capture(recorded_signals, name, retained, tmp_path):
    recorded_signals[name] = recorded_signals[name][retained]
    with pytest.raises(ValueError, match=f'{name}: expected one observation'):
        check_episode(tmp_path, 4, tmp_path, success_only=True)
    reference = tmp_path / 'reference'
    with pytest.raises(ValueError, match=f'{name}: expected one observation'):
        capture(output_dir=str(tmp_path), reference_dir=str(reference), seeds=[4])
    assert not reference.exists()


def test_complete_observations_and_sparse_commands_can_be_captured(recorded_signals, tmp_path):
    reference = tmp_path / 'reference'
    capture(output_dir=str(tmp_path), reference_dir=str(reference), seeds=[4])
    check_episode(tmp_path, 4, reference, success_only=False)


def test_failed_validation_leaves_capture_retryable(recorded_signals, monkeypatch, tmp_path):
    reference = tmp_path / 'reference'

    def invalid_second_seed(output, seed):
        if seed == 8:
            raise ValueError('incomplete recording')
        return EpisodeContainer(recorded_signals)

    with monkeypatch.context() as patch:
        patch.setattr(act_stack, 'read_episode', invalid_second_seed)
        with pytest.raises(ValueError, match='incomplete recording'):
            capture(output_dir=str(tmp_path), reference_dir=str(reference))
    assert not reference.exists()
    capture(output_dir=str(tmp_path), reference_dir=str(reference))
    for seed in (4, 8):
        check_episode(tmp_path, seed, reference, success_only=False)


def test_failed_write_leaves_capture_retryable(recorded_signals, monkeypatch, tmp_path):
    reference = tmp_path / 'reference'

    def fail_second_write(path, trace):
        write_npz(path, trace)
        if path.name == REFERENCE_FILENAME.format(seed=8):
            raise OSError('write failed')

    with monkeypatch.context() as patch:
        patch.setattr(act_stack, 'write_npz', fail_second_write)
        with pytest.raises(OSError, match='write failed'):
            capture(output_dir=str(tmp_path), reference_dir=str(reference))
    assert not reference.exists()
    capture(output_dir=str(tmp_path), reference_dir=str(reference))
    for seed in (4, 8):
        check_episode(tmp_path, seed, reference, success_only=False)


def test_capture_preserves_a_destination_created_during_writing(recorded_signals, monkeypatch, tmp_path):
    reference = tmp_path / 'reference'

    def concurrent_destination(path, trace):
        write_npz(path, trace)
        reference.mkdir()

    monkeypatch.setattr(act_stack, 'write_npz', concurrent_destination)
    with pytest.raises(FileExistsError):
        capture(output_dir=str(tmp_path), reference_dir=str(reference), seeds=[4])
    assert reference.is_dir() and not list(reference.iterdir())


def test_failed_publication_leaves_capture_retryable(recorded_signals, monkeypatch, tmp_path):
    reference = tmp_path / 'reference'

    def fail_rename(source, destination):
        raise OSError('rename failed')

    with monkeypatch.context() as patch:
        patch.setattr(type(reference), 'rename', fail_rename)
        with pytest.raises(OSError, match='rename failed'):
            capture(output_dir=str(tmp_path), reference_dir=str(reference), seeds=[4])
    assert not reference.exists()
    capture(output_dir=str(tmp_path), reference_dir=str(reference), seeds=[4])
    check_episode(tmp_path, 4, reference, success_only=False)


@pytest.mark.parametrize(
    ('height', 'green_x', 'finger_x', 'expected'),
    [(0.0299, 0, 0.1, True), (0.06, 0, 0.1, False), (0.0299, 0.1, 0.2, False), (0.0299, 0, 0.0249, False)],
)
def test_stacking_requires_cube_contact_and_release(height, green_x, finger_x, expected):
    model = mj.MjModel.from_xml_string(f'''
        <mujoco><worldbody>
          <body name="{STACK_RED_CUBE.body_name}" pos="0 0 0.01"><geom type="box" size="0.02 0.02 0.01"/></body>
          <body name="{STACK_GREEN_CUBE.body_name}" pos="{green_x} 0 {height}">
            <freejoint/><geom type="box" size="0.02 0.02 0.01"/>
          </body>
          <body name="{FINGER_BODIES[0]}" pos="{finger_x} 0 0.03">
            <geom type="box" size="0.005 0.005 0.005"/>
          </body>
        </worldbody></mujoco>
    ''')
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert (
        is_supported_stack(
            model,
            data,
            model.body(STACK_RED_CUBE.body_name).id,
            model.body(STACK_GREEN_CUBE.body_name).id,
            {model.body(FINGER_BODIES[0]).id},
        )
        == expected
    )


def test_brief_or_interrupted_support_does_not_count_as_success():
    times = np.arange(0, EPISODE_SECONDS * 1_000_000_000 + 1, 100_000_000)
    supported = np.zeros(len(times), dtype=bool)
    supported[10:15] = True
    supported[16:21] = True
    trace = {SUPPORTED + TIME_SUFFIX: times, SUPPORTED: supported}
    with pytest.raises(ValueError, match='never rested'):
        check_stacking(trace)
    supported[15] = True
    assert check_stacking(trace) == 1.5


def test_an_early_recording_cannot_pass_by_already_having_stacked():
    times = np.arange(0, 1_000_000_000, 100_000_000)
    with pytest.raises(ValueError, match='full episode'):
        check_stacking({SUPPORTED + TIME_SUFFIX: times, SUPPORTED: np.ones(len(times), dtype=bool)})


def test_stacking_uses_support_timestamps():
    trace = {
        CUBE_POSES + TIME_SUFFIX: np.array([0, 1_000_000_000, 2_000_000_000, 15_000_000_000]),
        SUPPORTED + TIME_SUFFIX: np.array([0, 250_000_000, 500_000_000, 15_000_000_000]),
        SUPPORTED: np.array([True, True, True, False]),
    }
    assert check_stacking(trace) == 0.5


def test_support_comparison_reports_its_own_timestamp():
    expected = {SUPPORTED: np.array([False, True]), SUPPORTED + TIME_SUFFIX: np.array([0, 2_000_000_000])}
    actual = {**expected, SUPPORTED: np.array([False, False])}
    with pytest.raises(ValueError, match=f'{SUPPORTED}: first difference at 2.000000000s'):
        compare_trace(actual, expected)


@pytest.mark.parametrize(
    'actual',
    [
        pytest.param(
            {CUBE_POSES: np.array([0.0, 0.001, 0.0]), CUBE_POSES + TIME_SUFFIX: np.array([0, 2, 4])}, id='value'
        ),
        pytest.param({CUBE_POSES: np.zeros(3), CUBE_POSES + TIME_SUFFIX: np.array([0, 3, 4])}, id='timestamp'),
        pytest.param({CUBE_POSES + TIME_SUFFIX: np.array([0, 2, 4])}, id='missing'),
        pytest.param({CUBE_POSES: np.zeros(2), CUBE_POSES + TIME_SUFFIX: np.array([0, 2, 4])}, id='truncated'),
        pytest.param(
            {CUBE_POSES: np.array([0.0, np.nan, 0.0]), CUBE_POSES + TIME_SUFFIX: np.array([0, 2, 4])}, id='nan'
        ),
    ],
)
def test_comparison_rejects_behavior_changes(actual):
    expected = {CUBE_POSES: np.zeros(3), CUBE_POSES + TIME_SUFFIX: np.array([0, 2, 4])}
    with pytest.raises(ValueError):
        compare_trace(actual, expected)


def test_identical_trace_passes():
    trace = {CUBE_POSES: np.zeros((2, 2, 7)), CUBE_POSES + TIME_SUFFIX: np.array([0, 2])}
    compare_trace(trace, trace)


def test_reference_capture_refuses_to_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        capture(reference_dir=str(tmp_path), output_dir='unused')


@pytest.mark.parametrize('url', ['localhost:8000?codec.fps=10', 'http://localhost:8000/api/v1/session/other'])
def test_url_cannot_change_the_pinned_pipeline(url):
    with pytest.raises(ValueError, match='server origin'):
        checkpoint_url(url)
