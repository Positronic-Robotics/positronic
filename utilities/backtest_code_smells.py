"""Evaluate lint candidates on historical fixes and tracked Python sources; write the evidence as JSON."""

import argparse
import ast
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

from utilities.check_code_smells import python_paths, repeated_control_blocks, repeated_guards

PROFILE = Path(__file__).with_name('code_smells.toml')
EXPLORATORY_RULES = 'SLF001,PLC2701,ARG001,ARG002,RET505,SIM102,SIM114,PLR0913,PLR0917'


def git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout


def lint(root: Path, paths: list[Path], *, source: str | None = None) -> list[dict]:
    command = [
        'ruff',
        'check',
        '--config',
        str(PROFILE),
        '--extend-select',
        EXPLORATORY_RULES,
        '--output-format',
        'json',
        '--no-cache',
    ]
    command += ['--stdin-filename', paths[0], '-'] if source is not None else paths
    result = subprocess.run(command, cwd=root, input=source, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or result.stdout)
    findings = json.loads(result.stdout)
    for finding in findings:
        finding['filename'] = Path(finding['filename']).relative_to(root)
        finding.pop('fix', None)
    return findings


@dataclass(frozen=True)
class Case:
    name: str
    commit: str
    path: Path
    symbol: str
    rule: str
    held_out: bool = False


CASES = (
    Case('Repeated finite-number validation', 'a77112635', Path('positronic/policy/processors.py'), '', 'QSM001'),
    Case(
        'Playback loop keeps too much state',
        'a77112635',
        Path('positronic/policy/processors.py'),
        'TrajectoryPlayer.run',
        'PLR0914',
    ),
    Case(
        'Observation helper has too many arguments',
        'a6ee8bab5',
        Path('positronic/policy/processors.py'),
        'InferenceSchedule._call_obs',
        'PLR0913',
    ),
    Case(
        'Call bookkeeping complicates the main loop',
        '38b5faa79',
        Path('positronic/policy/processors.py'),
        'InferenceSchedule.run',
        'PLR0912',
    ),
    Case(
        'Timing bookkeeping lengthens the main loop',
        'eb903787f',
        Path('positronic/policy/processors.py'),
        'InferenceSchedule.run',
        'PLR0915',
    ),
    Case(
        'Two inference submission paths',
        '126b77436',
        Path('positronic/policy/processors.py'),
        '_InferenceSchedule.run',
        'QSM002',
    ),
    Case(
        'Cryptic overlapping-image selection',
        '94c29e267',
        Path('positronic/policy/codecs/image.py'),
        'TransformImages.encode',
        '*',
    ),
    Case(
        'One component borrows another component representation',
        '880404c05',
        Path('positronic/policy/processors.py'),
        'RTCSchedule.run',
        'SLF001',
    ),
    Case('Consolidate RTC context assembly', '396a461fc', Path('positronic/policy/processors.py'), '', '*', True),
    Case('Combine adding and removing mapping keys', '2f9b02f48', Path('positronic/utils/lazy.py'), '', '*', True),
    Case('Remove redundant image metadata', '763afbc02', Path('positronic/policy/codecs/image.py'), '', '*', True),
    Case('Keep a private helper beside its user', 'b80805223', Path('positronic/policy/processors.py'), '', '*', True),
)


def symbol_span(source: str, symbol: str) -> tuple[int, int]:
    if not symbol:
        return 1, len(source.splitlines())
    scope = ast.parse(source)
    for name in symbol.split('.'):
        scope = next(
            node
            for node in scope.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
        )
    assert isinstance(scope, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and scope.end_lineno is not None
    return scope.lineno, scope.end_lineno


def historical_case(root: Path, case: Case) -> dict:
    commit = git(root, 'rev-parse', case.commit).strip()
    result = asdict(replace(case, commit=commit))
    for side, revision in [('before', commit + '^'), ('after', commit)]:
        source = git(root, 'show', f'{revision}:{case.path}')
        first, last = symbol_span(source, case.symbol)
        result[side] = [
            finding
            for finding in lint(root, [case.path], source=source)
            if first <= finding['location']['row'] <= last and (case.rule == '*' or finding['code'] == case.rule)
        ]
        for rule, check in [('QSM001', repeated_guards), ('QSM002', repeated_control_blocks)]:
            if case.rule in ('*', rule):
                result[side].extend(
                    {'code': rule, 'occurrences': [asdict(item) for item in group]}
                    for group in check({case.path: source})
                    if any(first <= item.line <= last for item in group)
                )
    result['outcome'] = (
        'cleared'
        if result['before'] and not result['after']
        else 'persists'
        if result['before'] and result['after']
        else 'introduced'
        if result['after']
        else 'missed'
    )
    return result


def _only_forwards_arguments(node: ast.FunctionDef) -> bool:
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if node.decorator_list or len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    call = body[0].value
    if not isinstance(call, ast.Call) or node.args.vararg or node.args.kwarg:
        return False
    parameters = [
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if arg.arg not in ('self', 'cls')
    ]
    arguments = [*call.args, *(arg.value for arg in call.keywords)]
    return (
        bool(parameters)
        and all(isinstance(arg, ast.Name) for arg in arguments)
        and sorted(parameters) == sorted(arg.id for arg in arguments if isinstance(arg, ast.Name))
    )


def forwarding_helpers(source: str) -> list[dict]:
    """Measure the deliberately narrow candidate 'private helper only forwards its arguments'."""
    return [
        {'line': node.lineno, 'symbol': node.name, 'body': ast.unparse(node.body[-1])}
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith('_')
        and not node.name.endswith('__')
        and _only_forwards_arguments(node)
    ]


def analyze_current_sources(root: Path) -> dict:
    paths = python_paths(root, include_tests=True)
    findings = lint(root, paths)
    production = set(python_paths(root))
    sources = {path: (root / path).read_text() for path in sorted(production)}
    return {
        'files': len(paths),
        'production_files': len(production),
        'production_counts': dict(Counter(f['code'] for f in findings if f['filename'] in production)),
        'test_counts': dict(Counter(f['code'] for f in findings if f['filename'] not in production)),
        'findings': findings,
        'repeated_guards': [[asdict(item) for item in group] for group in repeated_guards(sources)],
        'repeated_control_blocks': [[asdict(item) for item in group] for group in repeated_control_blocks(sources)],
        'forwarding_helpers': [
            {'path': path, **f} for path, source in sources.items() for f in forwarding_helpers(source)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='JSON evidence file')
    args = parser.parse_args()
    root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').strip())
    start = perf_counter()
    report = {
        'revision': git(root, 'rev-parse', 'HEAD').strip(),
        'working_tree': git(root, 'status', '--porcelain'),
        'ruff_version': subprocess.run(
            ['ruff', '--version'], check=True, capture_output=True, text=True
        ).stdout.strip(),
        'profile': PROFILE.read_text(),
        'limitations': [
            'Historical source is parsed, not executed; this does not run old application tests.',
            'Calibration cases informed candidate selection; they are not a blind accuracy estimate.',
            'Held-out cases were selected by commit subject before inspecting their diffs.',
            'Wildcard cases report all findings in their scope; inspect whether a change concerns the reviewed issue.',
            'Accepted current code is not automatically a false positive: every recommendation needs review.',
        ],
        'history': [historical_case(root, case) for case in CASES],
        'current': analyze_current_sources(root),
        'elapsed_sec': perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    for case in report['history']:
        split = 'held-out' if case['held_out'] else 'calibration'
        print(f'{split}: {case["name"]}: {case["outcome"]} ({len(case["before"])} -> {len(case["after"])})')
    print('Production:', report['current']['production_counts'])
    print('Tests:', report['current']['test_counts'])
    print(f'Evidence: {args.output}')


if __name__ == '__main__':
    main()
