from enum import IntEnum

import numpy as np
import pytest

from positronic.utils import check_positive_int


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
