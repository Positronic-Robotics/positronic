"""A driver waits by yielding, so no driver module blocks in the sleep of a system library."""

import ast
from pathlib import Path

import pytest

DRIVERS = Path(__file__).resolve().parent.parent
WAIVER = '# blocking-sleep-ok:'
BLOCKING = {('time', 'sleep')}


def _main_guard_lines(tree: ast.Module) -> set[int]:
    """Every line under an ``if __name__ == '__main__':`` block: there the module IS the runner."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if '__main__' in test and '__name__' in test:
            for body in node.body:
                lines.update(range(body.lineno, (body.end_lineno or body.lineno) + 1))
    return lines


def _blocking_calls(path: Path) -> list[int]:
    source = path.read_text()
    tree = ast.parse(source)
    exempt = _main_guard_lines(tree)
    waived = {n for n, line in enumerate(source.splitlines(), 1) if WAIVER in line}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if (node.func.value.id, node.func.attr) not in BLOCKING:
            continue
        # A waiver reads on the call's own line or the line above it, where a comment naturally sits.
        if node.lineno in exempt or waived & {node.lineno, node.lineno - 1}:
            continue
        found.append(node.lineno)
    return found


# A test that drives a control loop IS a runner, so it sleeps like one.
MODULES = sorted(p for p in DRIVERS.rglob('*.py') if 'tests' not in p.parts)


@pytest.mark.parametrize('path', MODULES, ids=lambda p: str(p.relative_to(DRIVERS)))
def test_a_driver_does_not_block_in_a_system_sleep(path: Path):
    """`yield pimm.Sleep(secs)` hands the wait to the runner, which advances the other control
    systems meanwhile; `time.sleep` takes the wait and stalls them. Exempt: a module's own
    `__main__` demo, which IS a runner, and a line carrying `# blocking-sleep-ok: <reason>`."""
    assert _blocking_calls(path) == [], (
        f'{path.relative_to(DRIVERS)} blocks in time.sleep at these lines; yield pimm.Sleep(secs) '
        f'instead, or mark the line "{WAIVER} <reason>" if this code is the runner'
    )
