import pytest
from eval_vocabulary.timeline import (
    HARNESS_WALL,
    HARNESS_WORLD,
    RECEIVED_WORLD,
    RECORDED_TIME,
    RECORDING_TIMELINE_ORDER,
    recording_timeline,
)


def test_the_timeline_names_are_what_recordings_carry():
    assert (HARNESS_WORLD, HARNESS_WALL, RECEIVED_WORLD, RECORDED_TIME) == (
        'harness.world',
        'harness.wall',
        'received.world',
        'recorded',
    )


def test_a_reader_tries_the_harness_then_the_receipt_then_the_legacy_timeline():
    assert RECORDING_TIMELINE_ORDER == ('harness.world', 'received.world', 'recorded')


@pytest.mark.parametrize(
    ('timelines', 'expected'),
    [
        (('recorded', 'received.world', 'harness.world'), 'harness.world'),
        (('recorded', 'received.world', 'emitted.world'), 'received.world'),
        (('recorded',), 'recorded'),
        (('emitted.world', 'capture'), 'recorded'),
    ],
)
def test_a_reader_takes_the_first_timeline_of_the_order_that_the_recording_has(timelines, expected):
    assert recording_timeline(timelines) == expected
