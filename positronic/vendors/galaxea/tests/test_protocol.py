import msgpack
import numpy as np
import pytest

from positronic.vendors.galaxea import protocol


@pytest.mark.parametrize('length', [1, 16, 32, 48])
def test_chunk_transport_preserves_every_step(length):
    arms = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
    grips = np.linspace(0, 1, length, dtype=np.float32).reshape(length, 1)
    response = protocol.chunk_response({protocol.RIGHT_ARM: arms, protocol.RIGHT_GRIPPER: grips}, 'pick towel')
    decoded = msgpack.unpackb(msgpack.packb(response))
    assert len(decoded[protocol.ACTIONS]) == length
    np.testing.assert_array_equal([step[protocol.RIGHT_ARM] for step in decoded[protocol.ACTIONS]], arms)
    np.testing.assert_array_equal([step[protocol.RIGHT_GRIPPER] for step in decoded[protocol.ACTIONS]], grips)
    assert decoded[protocol.COT_TEXT] == 'pick towel'


def test_omitted_gripper_remains_omitted_on_every_step():
    response = protocol.chunk_response({protocol.RIGHT_ARM: np.zeros((32, 7))}, None)
    assert len(response[protocol.ACTIONS]) == 32
    assert all(protocol.RIGHT_GRIPPER not in step for step in response[protocol.ACTIONS])


@pytest.mark.parametrize('arms', [np.zeros((0, 7)), np.zeros((1, 32, 7)), np.zeros((32, 6)), np.full((32, 7), np.nan)])
def test_invalid_arm_chunk_fails(arms):
    with pytest.raises(ValueError):
        protocol.chunk_response({protocol.RIGHT_ARM: arms}, None)


def test_mismatched_part_lengths_fail_instead_of_repeating_steps():
    with pytest.raises(ValueError, match='Invalid chunk'):
        protocol.chunk_response(
            {protocol.RIGHT_ARM: np.zeros((32, 7)), protocol.RIGHT_GRIPPER: np.zeros((16, 1))}, None
        )
