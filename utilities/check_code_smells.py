"""Find repeated validation and control-path blocks without importing the inspected code."""

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from textwrap import indent


def python_paths(root: Path, *, include_tests: bool = False) -> list[Path]:
    """Use tracked Python sources, excluding generated stubs and optionally tests."""
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '-z', '--', '*.py'], check=True, capture_output=True, text=True
    )
    paths = [Path(name) for name in result.stdout.split('\0') if name]
    return [
        path
        for path in paths
        if 'stubs' not in path.parts
        and (
            include_tests or not ('tests' in path.parts or path.stem.startswith('test_') or path.stem.endswith('_test'))
        )
        and (root / path).is_file()
    ]


@dataclass(frozen=True)
class Occurrence:
    path: str
    line: int
    expression: str


def _scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        yield child
        yield from _scope_nodes(child)


class _Shape(ast.NodeTransformer):
    """Keep predicate values when bindings are supplied; otherwise compare operation structure only."""

    def __init__(self, bindings: set[str] | None = None):
        self._bindings = bindings
        self._names: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if self._bindings is None:
            return ast.Name(id='value', ctx=node.ctx)
        if node.id not in self._bindings:
            return node
        return ast.Name(id=self._names.setdefault(node.id, f'value_{len(self._names)}'), ctx=node.ctx)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant | ast.Name:
        return ast.Name(id='value', ctx=ast.Load()) if self._bindings is None else node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        return ast.Call(
            func=node.func,
            args=[self.visit(arg) for arg in node.args],
            keywords=[ast.keyword(arg=arg.arg, value=self.visit(arg.value)) for arg in node.keywords],
        )


class _Guards(ast.NodeVisitor):
    def __init__(self, path: str):
        self._path = path
        self._bindings: set[str] = set()
        self.groups: dict[tuple[str, str], list[Occurrence]] = defaultdict(list)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self._bindings
        self._bindings = previous | {
            item.arg if isinstance(item, ast.arg) else item.id
            for item in _scope_nodes(node)
            if isinstance(item, ast.arg) or isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        }
        self.generic_visit(node)
        self._bindings = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        if (
            isinstance(node.test, ast.BoolOp)
            and not node.orelse
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and isinstance(node.body[0].exc, ast.Call)
            and node.body[0].cause is None
        ):
            predicate = _Shape(self._bindings - {'self', 'cls'}).visit(deepcopy(node.test))
            key = (ast.dump(predicate), ast.dump(node.body[0].exc.func))
            self.groups[key].append(Occurrence(self._path, node.lineno, ast.unparse(node.test)))
        self.generic_visit(node)


def repeated_guards(sources: Mapping[str, str], *, minimum: int = 3) -> list[list[Occurrence]]:
    """Group guards with the same predicate shape and exception type; ignore diagnostic wording.

    Local variable spelling is ignored. Operators, literals, callable and attribute names stay significant.
    Matches suggest a shared validation helper; they do not establish semantic equivalence or a safe dependency.
    """
    if minimum < 2:
        raise ValueError('minimum must be at least two')
    groups: dict[tuple[str, str], list[Occurrence]] = defaultdict(list)
    for path, source in sorted(sources.items()):
        visitor = _Guards(path)
        visitor.visit(ast.parse(source, filename=path))
        for key, occurrences in visitor.groups.items():
            groups[key].extend(occurrences)
    return [occurrences for occurrences in groups.values() if len(occurrences) >= minimum]


def _candidate_blocks(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[tuple[int, ast.Module]]:
    for parent in (function, *_scope_nodes(function)):
        for _, body in ast.iter_fields(parent):
            if not isinstance(body, list):
                continue
            for first, second in pairwise(body):
                if not all(
                    isinstance(node, ast.Assign | ast.AnnAssign | ast.Expr | ast.Return) for node in (first, second)
                ):
                    continue
                block = ast.Module(body=[first, second], type_ignores=[])
                if sum(isinstance(node, ast.Call) for node in ast.walk(block)) >= 2:
                    yield id(body), block


def repeated_control_blocks(sources: Mapping[str, str]) -> list[list[Occurrence]]:
    """Find two-statement blocks repeated in separate control paths within a function.

    At least two calls must occur in each block. Values may differ; callable paths, operators and
    attribute names must match. This suggests duplication, not interchangeable behavior or a safe rewrite.
    """
    findings = []
    for path, source in sorted(sources.items()):
        tree = ast.parse(source, filename=path)
        functions = (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef))
        for function in functions:
            groups: dict[str, list[tuple[int, Occurrence]]] = defaultdict(list)
            for owner, block in _candidate_blocks(function):
                shape = ast.dump(_Shape().visit(deepcopy(block)))
                groups[shape].append((owner, Occurrence(path, block.body[0].lineno, ast.unparse(block))))
            findings.extend(
                [item for _, item in group] for group in groups.values() if len({owner for owner, _ in group}) > 1
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path, help='files to compare; defaults to tracked production Python')
    parser.add_argument('--include-tests', action='store_true', help='include tests in the default file selection')
    parser.add_argument('--minimum', type=int, default=3, help='minimum occurrences (default: 3)')
    parser.add_argument('--json', action='store_true', help='emit groups with every source location')
    args = parser.parse_args()
    if args.minimum < 2:
        parser.error('--minimum must be at least two')
    paths = args.paths or python_paths(Path.cwd(), include_tests=args.include_tests)
    sources = {str(path): path.read_text() for path in paths}
    findings = [
        {'rule': rule, 'occurrences': [asdict(item) for item in group]}
        for rule, groups in (
            ('QSM001', repeated_guards(sources, minimum=args.minimum)),
            ('QSM002', repeated_control_blocks(sources)),
        )
        for group in groups
    ]
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        for finding in findings:
            print(f'{finding["rule"]}: similar code in {len(finding["occurrences"])} places; inspect before extracting')
            for item in finding['occurrences']:
                print(f'  {item["path"]}:{item["line"]}:')
                print(indent(item['expression'], '    '))
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
