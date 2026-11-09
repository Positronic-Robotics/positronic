from enum import IntEnum

import numpy as np
import pytest

from positronic.utils import check_positive_int, check_rgb_frame


@pytest.mark.parametrize('value', [1, 2, 2**100])
def test_positive_int_returns_the_value(value):
    assert check_positive_int(value) == value


class _IntEnum(IntEnum):
    ONE = 1


@pytest.mark.parametrize('value', [0, -1, True, False, 1.0, '1', None, [], {}, np.int64(1), _IntEnum.ONE])
def test_positive_int_rejects_nonpositive_values_and_other_types(value):
    with pytest.raises(ValueError, match='value must be a positive integer'):
        check_positive_int(value)


def test_positive_int_names_the_invalid_argument():
    with pytest.raises(ValueError, match='count must be a positive integer, got 0'):
        check_positive_int(0, 'count')


@pytest.mark.parametrize('shape', [(2, 4, 3), (0, 4, 3), (2, 0, 3)])
@pytest.mark.parametrize('dtype', [np.uint8, np.float32])
def test_rgb_frame_accepts_any_height_width_and_dtype(shape, dtype):
    check_rgb_frame(np.empty(shape, dtype=dtype))


@pytest.mark.parametrize('shape', [(), (3,), (2, 4), (2, 4, 1), (2, 4, 4), (1, 2, 4, 3)])
def test_rgb_frame_rejects_other_shapes_and_names_the_field(shape):
    with pytest.raises(ValueError, match='Expected wrist shape'):
        check_rgb_frame(np.empty(shape), 'wrist')
