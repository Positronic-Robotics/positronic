import pytest

from pimm.time import RECEIVED_WORLD
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.eval import keys as eval_keys
from utilities.release_phail import _duration, verify_inference


@pytest.mark.parametrize(
    ('timelines', 'seconds'),
    [
        ((RECORDED_TIME,), 30),
        ((RECEIVED_WORLD,), 20),
        ((HARNESS_WORLD,), 10),
        ((RECORDED_TIME, RECEIVED_WORLD), 20),
        ((RECORDED_TIME, RECEIVED_WORLD, HARNESS_WORLD), 10),
    ],
)
def test_duration_uses_the_recordings_clock(timelines, seconds):
    finish = {HARNESS_WORLD: 10_000_000_000, RECEIVED_WORLD: 20_000_000_000, RECORDED_TIME: 30_000_000_000}
    signal = DummySignal([[0] * len(timelines), [finish[name] for name in timelines]], [0, 1], timelines=timelines)
    assert _duration(EpisodeContainer({'state': signal})) == seconds


def test_explicit_duration_does_not_require_signals():
    assert _duration(EpisodeContainer({eval_keys.DURATION: 12.5})) == 12.5


def test_missing_duration_and_clock_is_an_error():
    with pytest.raises(ValueError, match='no signals'):
        _duration(EpisodeContainer({}))


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_verification_checks_outliers_without_duration_metadata(timeline):
    episodes = [
        EpisodeContainer({
            'model': 'act',
            'eval.object': 'cube',
            'eval.successful_items': count,
            'eval.total_items': count,
            'state': DummySignal([0, 1_000_000_000], [0, 1], timelines=(timeline,)),
        })
        for count in [1] * 10 + [100]
    ]
    with pytest.raises(SystemExit, match='Verification failed'):
        verify_inference(dataset=episodes, force=True)
