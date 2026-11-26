"""Site changes to the MolmoSpaces episodes that a server runs, loaded from a file.

This module must work in MolmoSpaces' isolated interpreter without Positronic installed.

A hook file defines ``HOOK``, an ``EnvHook``. The server's interpreter has no positronic, so the file imports the base
class as ``from env_hook import EnvHook``. Each reset token names its hook file, or none for the default hook. The
server runs a hook file at the first episode that names its digest, after MolmoSpaces has loaded, and keeps the hook
for each later episode with that digest.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, NamedTuple

if __package__:
    from positronic.simulator.molmo_spaces import mapping
else:
    import mapping

HOOK_NAME = 'HOOK'


class EnvHook:
    """The default hook: it changes nothing.

    A hook can change the robot only within the bundled FR3 model: its seven arm joints, its kinematics and its 2F-85
    grasp site stay, for example a mount or a gripper detail changes. The eval records the bundled robot model.

    One server runs episodes of several hooks, one episode at a time. A hook changes only the config and the task of
    its own episode, because the server builds the model, the robot and the renderer of each episode again.
    """

    def eval_config(self, cfg: Any, episode: Any) -> None:
        """Change the MolmoSpaces eval config ``cfg`` of a copy of ``episode``, before the server uses ``cfg``.

        The server calls this once to list the episode's horizon and again to build its task, so the changes must
        depend on ``episode`` alone: no state kept between calls and no random draw. After this call, the server sets
        ``cfg.seed`` and ``cfg.task_horizon``.
        """

    def task_sampled(self, task: Any) -> None:
        """Change the sampled MolmoSpaces task before its first reset.

        After this call, the server drops the object image points sensor where the reset asks, and checks the arm
        joints and the grasp site.
        """

    def grip(self, gripper_qpos: Any) -> float:
        """The grip observation for the joint positions of the gripper move group."""
        return mapping.normalize_grip_qpos(gripper_qpos)

    def grip_actuator(self, grip_command: float) -> float:
        """The 2F-85 actuator command (0 to 255) for the policy's grip command.

        A hook that reports the grip on another scale in ``grip`` maps the command from that scale here.
        """
        return mapping.grip_command_to_actuator(grip_command)


class LoadedHook(NamedTuple):
    hook: EnvHook
    record: str | None  # The ``mapping.TASK_ENV_HOOK`` value of the hook: None for the default hook.


def load(path: Path | None, sha256: str | None = None) -> LoadedHook:
    """The ``HOOK`` of the file at ``path``, or the default hook when ``path`` is None.

    With ``sha256``, the file must have that digest, and the server runs the bytes it checked.
    """
    if path is None:
        return LoadedHook(EnvHook(), None)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'{path} is not a Python file')
    if spec.name in sys.modules:
        raise ValueError(f'{path} has the name of the loaded module {spec.name}; rename the hook file')
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f'{path} changed after the eval recorded its digest {sha256}')
    module = importlib.util.module_from_spec(spec)
    # A dataclass finds the globals of its module in sys.modules.
    sys.modules[spec.name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    hook = getattr(module, HOOK_NAME, None)
    if not isinstance(hook, EnvHook):
        raise ValueError(f'{path} must define {HOOK_NAME} as an EnvHook, not {type(hook).__name__}')
    return LoadedHook(hook, mapping.HookFile(str(path), digest).record)


class Hooks:
    """The hooks of a server's episodes: each hook file loads once for each digest, and the server keeps it.

    A server loads one hook file of each name, so two versions of one hook never share the modules they import.
    """

    def __init__(self) -> None:
        self._default = load(None)
        self._loaded: dict[str, LoadedHook] = {}

    def of(self, request: dict[str, Any]) -> LoadedHook:
        """The hook that a selection or a reset token names, or the default hook where it names none."""
        hook_file = mapping.HookFile.of(request)
        if hook_file is None:
            return self._default
        if hook_file.env_hook_sha256 not in self._loaded:
            self._loaded[hook_file.env_hook_sha256] = load(Path(hook_file.env_hook), hook_file.env_hook_sha256)
        return self._loaded[hook_file.env_hook_sha256]
