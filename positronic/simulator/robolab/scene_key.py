"""What a reset of the RoboLab env server builds the scene from.

RoboLab's interpreter imports this module as a top-level ``scene_key``, so it uses only Python.
"""

from typing import NamedTuple

if __package__:
    from positronic.simulator.robolab import keys
else:
    import keys  # pyright: ignore[reportMissingImports]


class SceneKey(NamedTuple):
    """A reset with the key of the built scene reuses that scene; a reset with another key builds a new one."""

    task: str
    instruction_type: str
    episode_length_s: float | None
    cameras: str
    num_envs: int


def scene_key(token: dict, num_envs: int) -> SceneKey:
    """The ``SceneKey`` of a reset with ``token`` on ``num_envs`` clones."""
    return SceneKey(
        task=token[keys.TOKEN_TASK],
        instruction_type=token[keys.TOKEN_INSTRUCTION_TYPE],
        episode_length_s=token.get(keys.TOKEN_EPISODE_LENGTH_S),
        cameras=token[keys.TOKEN_CAMERAS],
        num_envs=num_envs,
    )
