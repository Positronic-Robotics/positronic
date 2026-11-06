import json
import subprocess
import sys
from pathlib import Path

import pytest

from utilities.check_code_smells import python_paths, repeated_control_blocks, repeated_guards


def guards(*predicates: str, exception: str = 'ValueError') -> dict[str, str]:
    return {
        f'package_{index}/checks.py': (
            f'def validate(value, other):\n    if {predicate}:\n        raise {exception}("bad input {index}")\n'
        )
        for index, predicate in enumerate(predicates)
    }


def test_reports_all_locations_despite_parameter_names_and_diagnostics():
    sources = {
        'a.py': 'def f(rate):\n    if not isfinite(rate) or rate < 0:\n        raise ValueError("rate")\n',
        'b.py': 'def g(delay):\n    if not isfinite(delay) or delay < 0:\n        raise ValueError(f"{delay=}")\n',
        'c.py': 'def h(age):\n    if not isfinite(age) or age < 0:\n        raise ValueError("age")\n',
    }
    groups = repeated_guards(sources)
    assert len(groups) == 1
    assert [(item.path, item.line) for item in groups[0]] == [('a.py', 2), ('b.py', 2), ('c.py', 2)]
    assert groups[0][0].expression == 'not isfinite(rate) or rate < 0'


@pytest.mark.parametrize(
    'different',
    [
        'not isfinite(value) or value <= 0',
        'not isfinite(value) or value < 1',
        'not isfinite(value) and value < 0',
        'not isfinite(other) or value < 0',
        'not isnan(value) or value < 0',
    ],
)
def test_operators_literals_variable_relationships_and_predicates_matter(different):
    assert not repeated_guards(guards('not isfinite(value) or value < 0', different), minimum=2)


def test_distinct_types_and_exception_types_are_not_merged():
    assert not repeated_guards(
        guards('not isinstance(value, str) or not value', 'not isinstance(value, list) or not value'), minimum=2
    )
    sources = guards('not value or value < 0', 'not value or value < 0')
    sources['package_1/checks.py'] = sources['package_1/checks.py'].replace('ValueError', 'TypeError')
    assert not repeated_guards(sources, minimum=2)


def test_small_guards_and_single_use_helpers_are_not_reported():
    assert not repeated_guards(guards('value < 0', 'value < 0', 'value < 0'))
    assert not repeated_guards(guards('not isfinite(value) or value < 0'))


@pytest.mark.parametrize('extra', ['        log(value)\n', '    else:\n        accept(value)\n'])
def test_guards_with_additional_behavior_are_not_equated_with_validation(extra):
    sources = guards('not value or value < 0', 'not value or value < 0')
    sources['package_1/checks.py'] += extra
    assert not repeated_guards(sources, minimum=2)


def test_nested_function_bindings_do_not_erase_outer_global_type_names():
    source = """
def outer(value):
    def nested(str):
        return str
    if not isinstance(value, str) or not value:
        raise ValueError('str')
    if not isinstance(value, list) or not value:
        raise ValueError('list')
"""
    assert not repeated_guards({'types.py': source}, minimum=2)


def test_local_variable_renaming_and_async_functions_are_supported():
    source = """
async def first():
    width = await read()
    if not isfinite(width) or width < 0:
        raise ValueError('width')

async def second():
    height = await read()
    if not isfinite(height) or height < 0:
        raise ValueError('height')
"""
    assert len(repeated_guards({'async.py': source}, minimum=2)) == 1


def test_syntax_errors_and_invalid_thresholds_fail_visibly():
    with pytest.raises(SyntaxError):
        repeated_guards({'broken.py': 'def broken('})
    with pytest.raises(ValueError, match='at least two'):
        repeated_guards({}, minimum=1)


def test_shared_validator_removes_the_finding():
    sources = guards(
        'not isfinite(value) or value < 0', 'not isfinite(value) or value < 0', 'not isfinite(value) or value < 0'
    )
    assert repeated_guards(sources)
    helper = 'def check(value):\n    if not isfinite(value) or value < 0:\n        raise ValueError("value")\n'
    assert not repeated_guards({'helper.py': helper, 'callers.py': 'check(rate)\ncheck(delay)\ncheck(age)\n'})


def test_cli_compares_files_and_returns_a_lint_exit_status(tmp_path):
    sources = guards(
        'not isfinite(value) or value < 0', 'not isfinite(value) or value < 0', 'not isfinite(value) or value < 0'
    )
    paths = []
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir()
        path.write_text(source)
        paths.append(str(path))
    script = Path(__file__).resolve().parents[1] / 'check_code_smells.py'
    result = subprocess.run(
        [sys.executable, str(script), '--json', *paths], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 1, result.stderr
    findings = json.loads(result.stdout)
    assert findings[0]['rule'] == 'QSM001'
    assert {item['path'] for item in findings[0]['occurrences']} == set(paths)
    clean = subprocess.run([sys.executable, str(script), paths[0]], cwd=tmp_path, text=True, capture_output=True)
    assert clean.returncode == 0, clean.stderr


def test_repeated_blocks_are_generic_and_allow_different_values():
    source = """
def process(service, value, retry):
    if not retry:
        started = clock()
        result = service.send(encode(None))
    value = transform(value)
    if retry:
        started = clock()
        result = service.send(encode(value))
    return result
"""
    groups = repeated_control_blocks({'service.py': source})
    assert len(groups) == 1
    assert [item.line for item in groups[0]] == [4, 8]


@pytest.mark.parametrize('different', ['decode(value)', 'encode(value) + 1', 'service.send(value)'])
def test_distinct_operations_are_not_reported_as_the_same_block(different):
    source = f"""
def process(service, value, retry):
    if retry:
        started = clock()
        result = service.send(encode(value))
    else:
        started = clock()
        result = service.send({different})
"""
    assert not repeated_control_blocks({'service.py': source})


def test_parallel_calls_and_repetitive_declarations_are_not_duplicate_control_paths():
    source = """
def fanout(runtime, first, second, obs):
    left = runtime.submit(first, obs)
    right = runtime.submit(second, obs)
    return left, right

def options(parser):
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--mode')
    parser.add_argument('--format')
"""
    assert not repeated_control_blocks({'parallel.py': source})


def test_nested_functions_are_compared_independently():
    source = """
def outer(service):
    def inner():
        started = clock()
        result = service.send(encode(1))
    started = clock()
    result = service.send(encode(1))
"""
    assert not repeated_control_blocks({'nested.py': source})


def test_default_scope_is_tracked_python_and_tests_are_opt_in(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    names = ['driver.py', 'other package/api.py', 'package/tests/test_api.py', 'stubs/generated.py']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '--', *names], check=True)
    (tmp_path / 'private_experiment.py').write_text('')
    assert set(python_paths(tmp_path)) == {Path('driver.py'), Path('other package/api.py')}
    assert set(python_paths(tmp_path, include_tests=True)) == {
        Path('driver.py'),
        Path('other package/api.py'),
        Path('package/tests/test_api.py'),
    }
