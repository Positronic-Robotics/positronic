"""Site changes to the MolmoSpaces episodes that a server runs, loaded from a file.

This module must work in MolmoSpaces' isolated interpreter without Positronic installed.

A hook file defines ``HOOK``, an ``EnvHook``. The server runs the file before it loads MolmoSpaces, so the file's
top level can choose the asset packs that MolmoSpaces fetches when it loads.
"""

import importlib.util
from pathlib import Path
from typing import Any

if __package__:
    from positronic.simulator.molmo_spaces import mapping
else:
    import mapping

HOOK_NAME = 'HOOK'
# The server's flag that names the hook file.
HOOK_FLAG = '--hook'


class EnvHook:
    """The default hook: it changes nothing."""

    def eval_config(self, cfg: Any, episode: Any) -> None:
        """Change the MolmoSpaces eval config ``cfg`` of ``episode`` before the server samples its task."""

    def task_sampled(self, task: Any) -> None:
        """Change the sampled MolmoSpaces task before its first reset."""

    def grip(self, gripper_qpos: Any) -> float:
        """The grip observation in [0, 1] for the joint positions of the gripper move group."""
        return mapping.normalize_grip_qpos(gripper_qpos)


def load(path: Path | None) -> EnvHook:
    """The ``HOOK`` of the file at ``path``, or the default hook when ``path`` is None."""
    if path is None:
        return EnvHook()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'{path} is not a Python file')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hook = getattr(module, HOOK_NAME, None)
    if not isinstance(hook, EnvHook):
        raise ValueError(f'{path} must define {HOOK_NAME} as an EnvHook, not {type(hook).__name__}')
    return hook
