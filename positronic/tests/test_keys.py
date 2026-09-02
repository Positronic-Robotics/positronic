import ast
from pathlib import Path

from positronic import keys
from positronic.eval import keys as eval_keys
from positronic.simulator.libero import keys as libero_keys
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.robolab import keys as robolab_keys

# Namespaced raw wire keys that denote an observation signal, and the keys a trial records — the params it
# runs under and the verdict it ends on — wherever they appear as a string literal. Writing any of these as a
# bare literal instead of importing the constant is what this guard forbids — the value must live once, in
# the module that owns it, so a rename stays a single-site change.
_GUARDED = {
    keys.JOINTS,
    keys.JOINT_VEL,
    keys.EE_POSE,
    keys.WRIST_IMAGE,
    keys.EXTERIOR_IMAGE,
    eval_keys.SUCCESS,
    eval_keys.TERMINATED,
    eval_keys.CHARGE_INFERENCE_TIME,
    eval_keys.UNIVERSE,
    eval_keys.EMBODIMENT,
    eval_keys.TIMEOUT,
    robolab_keys.EPISODE_LENGTH,
    eval_keys.SEED,
    eval_keys.TRIAL_INDEX,
    eval_keys.TRIAL_COUNT,
    libero_keys.SUITE,
    libero_keys.TASK_ID,
    libero_keys.CAMERA_RESOLUTION,
    libero_keys.CONTROL_MODE,
    libero_keys.SETTLE_STEPS,
    eval_keys.TASK,
    robolab_keys.INSTRUCTION_TYPE,
    molmo_keys.EPISODE_INDEX,
    molmo_keys.TASK_HORIZON,
    keys.OBS_TIME_NS,
    keys.WALL_TIME_NS,
}
# keys.GRIP and keys.TASK are deliberately not guarded: their values are bare tokens the wire reuses
# across unrelated namespaces (action-command grip, vendor state-vectors, scene/reset tokens), so a
# literal-value match cannot tell the observation key from those and would fire on legitimate code.

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent
# Every ``keys.py`` defines the keys of its package; it is the one place a value is spelled.
_KEY_MODULES = sorted(_PACKAGE_ROOT.rglob('keys.py'))
# First-party trees whose Python consumes the wire and must import the constants rather than
# re-spell the literals — the package itself and the repo's `utilities/` scripts.
_GUARDED_ROOTS = (_PACKAGE_ROOT, _REPO_ROOT / 'utilities')


def _str_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, node.lineno


def test_no_guarded_key_literals():
    offenders = []
    for root in _GUARDED_ROOTS:
        for path in sorted(root.rglob('*.py')):
            if path in _KEY_MODULES:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for value, lineno in _str_literals(tree):
                if value in _GUARDED:
                    offenders.append(f'{path.relative_to(_REPO_ROOT)}:{lineno}: {value!r}')
    listing = '\n'.join(offenders)
    assert not offenders, f'Guarded key literals found — import the constant from its module:\n{listing}'


def test_keys_modules_import_nothing():
    # A keys module must stay a dependency-free leaf so an out-of-repo consumer can depend on it alone,
    # without dragging in the rest of positronic (or its optional torch/lerobot deps). Any import statement
    # appearing in one breaks that contract.
    imports = [
        f'{path.relative_to(_REPO_ROOT)}:{node.lineno}'
        for path in _KEY_MODULES
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not imports, 'A keys module must import nothing (dependency-free leaf module):\n' + '\n'.join(imports)
