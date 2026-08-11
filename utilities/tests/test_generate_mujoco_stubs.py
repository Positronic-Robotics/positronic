import pytest

from utilities import generate_mujoco_stubs as gms


def test_machine_specific_constants_keep_their_type_and_lose_their_value():
    text = "HEADERS_DIR: str = '/home/someone/.venv/lib/python3.13/site-packages/mujoco/include'\n"
    assert gms._normalise(text) == 'HEADERS_DIR: str\n'


def test_machine_specific_value_comments_are_dropped():
    text = "PLUGIN_HANDLES: list  # value = [<CDLL '/home/someone/libsensor.so', handle 71a at 0x72e>]\n"
    assert gms._normalise(text) == 'PLUGIN_HANDLES: list\n'


def test_other_constants_keep_their_value():
    text = 'mjMAXCONPAIR: int = 50\n'
    assert gms._normalise(text) == text


def test_indented_attribute_sharing_a_name_is_untouched():
    text = '    _SYSTEM: str = "Linux"\n'
    assert gms._normalise(text) == text


def test_trailing_whitespace_goes_and_the_file_ends_on_one_newline():
    assert gms._normalise('def f() -> int:   \n    ...\n\n\n') == 'def f() -> int:\n    ...\n'


def test_a_default_before_a_required_parameter_is_dropped():
    line = '    def f(degree: bool, sequence: MjCharVec = None, orientation: MjsOrientation) -> None:'
    assert gms._drop_unusable_defaults(line) == (
        '    def f(degree: bool, sequence: MjCharVec, orientation: MjsOrientation) -> None:'
    )


def test_a_default_with_only_defaults_after_it_is_kept():
    line = '    def f(self, a: int = 1, b: list = [-1.0, 2.0]) -> None:'
    assert gms._drop_unusable_defaults(line) == line


def test_a_keyword_only_default_before_a_required_parameter_is_kept():
    line = 'def f(a: int, *, b: int = 1, c: int) -> None:'
    assert gms._drop_unusable_defaults(line) == line


def test_commas_inside_annotations_do_not_split_parameters():
    line = 'def f(a: typing.Annotated[list[float], "FixedSize(4)"] = None, b: int) -> None:'
    assert gms._drop_unusable_defaults(line) == (
        'def f(a: typing.Annotated[list[float], "FixedSize(4)"], b: int) -> None:'
    )


def test_an_attribute_named_after_a_keyword_is_dropped_and_its_alias_kept():
    text = 'class MjVisual:\n    global: Global\n    global_: Global\n'
    assert gms._parseable(text, 'x.pyi') == 'class MjVisual:\n    global_: Global\n'


def test_a_construct_the_grammar_rejects_and_no_rule_covers_raises():
    with pytest.raises(SyntaxError):
        gms._parseable('def f(:\n', 'x.pyi')
