import ast
from pathlib import Path

from positronic import keys

# Namespaced raw wire keys that denote an observation signal wherever they appear as a string literal.
# Writing any of these as a bare literal instead of importing the constant is what this guard forbids —
# the value must live once, in `positronic.keys`, so a rename stays a single-site change.
_GUARDED = {keys.JOINTS, keys.JOINT_VEL, keys.EE_POSE, keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE}
# keys.GRIP and keys.TASK are deliberately not guarded: their values are bare tokens the wire reuses
# across unrelated namespaces (action-command grip, vendor state-vectors, scene/reset tokens), so a
# literal-value match cannot tell the observation key from those and would fire on legitimate code.

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_KEYS_MODULE = _PACKAGE_ROOT / 'keys.py'


def _str_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, node.lineno


def test_no_raw_observation_key_literals():
    offenders = []
    for path in sorted(_PACKAGE_ROOT.rglob('*.py')):
        if path == _KEYS_MODULE:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for value, lineno in _str_literals(tree):
            if value in _GUARDED:
                offenders.append(f'{path.relative_to(_PACKAGE_ROOT.parent)}:{lineno}: {value!r}')
    assert not offenders, (
        'Raw observation-key literals found — import the constant from `positronic.keys`:\n' + '\n'.join(offenders)
    )
