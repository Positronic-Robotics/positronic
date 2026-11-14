"""Site changes to the MolmoSpaces episodes that a server runs, loaded from a file.

This module must work in MolmoSpaces' isolated interpreter without Positronic installed.

A hook file defines ``HOOK``, an ``EnvHook``. The server's interpreter has no positronic, so the file imports the base
class as ``from env_hook import EnvHook``. The server runs the file before it loads MolmoSpaces, so the file's top level
can choose the asset packs that MolmoSpaces fetches when it loads.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

if __package__:
    from positronic.simulator.molmo_spaces import mapping
else:
    import mapping

HOOK_NAME = 'HOOK'


class EnvHook:
    """The default hook: it changes nothing.

    A hook can change the robot only within the bundled FR3 model: its seven arm joints, its kinematics and its 2F-85
    grasp site stay, for example a mount or a gripper detail changes. The eval records the bundled robot model.
    """

    def eval_config(self, cfg: Any, episode: Any) -> None:
        """Change the MolmoSpaces eval config ``cfg`` of a copy of ``episode``, before the server uses ``cfg``.

        The server lists each episode's horizon and builds its task with this config. After this call, it sets
        ``cfg.seed`` and ``cfg.task_horizon``.
        """

    def task_sampled(self, task: Any) -> None:
        """Change the sampled MolmoSpaces task before its first reset.

        After this call, the server drops the object image points sensor where the reset asks, and checks the arm
        joints and the grasp site.
        """

    def grip(self, gripper_qpos: Any) -> float:
        """The grip observation for the joint positions of the gripper move group.

        It maps the observation only: the server sends the grip command to the 2F-85 actuator as MolmoSpaces does.
        """
        return mapping.normalize_grip_qpos(gripper_qpos)


def load(path: Path | None, sha256: str | None = None) -> EnvHook:
    """The ``HOOK`` of the file at ``path``, or the default hook when ``path`` is None.

    With ``sha256``, the file must have that digest, and the server runs the bytes it checked.
    """
    if path is None:
        return EnvHook()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'{path} is not a Python file')
    if spec.name in sys.modules:
        raise ValueError(f'{path} has the name of the loaded module {spec.name}; rename the hook file')
    source = path.read_bytes()
    if sha256 is not None and hashlib.sha256(source).hexdigest() != sha256:
        raise ValueError(f'{path} changed after the eval recorded its digest {sha256}')
    module = importlib.util.module_from_spec(spec)
    # A dataclass finds the globals of its module in sys.modules.
    sys.modules[spec.name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    hook = getattr(module, HOOK_NAME, None)
    if not isinstance(hook, EnvHook):
        raise ValueError(f'{path} must define {HOOK_NAME} as an EnvHook, not {type(hook).__name__}')
    return hook
