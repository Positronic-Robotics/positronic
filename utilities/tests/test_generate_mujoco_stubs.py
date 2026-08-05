import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_mujoco_stubs as gms  # noqa: E402


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
