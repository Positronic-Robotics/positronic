import re

import pytest
from platform_client.boards import BoardRef
from pydantic import TypeAdapter, ValidationError


def test_a_board_slug_is_a_str_carrying_its_type():
    board = BoardRef('nebius-2026/robolab/public_subset')
    assert board == 'nebius-2026/robolab/public_subset'
    assert isinstance(board, BoardRef)


@pytest.mark.parametrize('value', ['', ' ', 'two words'])
def test_a_value_that_could_never_be_a_board_slug_is_refused(value: str):
    with pytest.raises(ValueError, match=f'^not a board slug: {re.escape(repr(value))}$'):
        BoardRef(value)


def test_a_model_refuses_a_blank_board_slug_in_the_same_words():
    with pytest.raises(ValidationError, match="not a board slug: ''"):
        TypeAdapter(BoardRef).validate_python('')
