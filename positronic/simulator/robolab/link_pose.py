"""The pose of a RoboLab contact object in each clone frame, for the ``gt_state`` of ``env.py``.

RoboLab's interpreter imports this module as a top-level ``link_pose``, so it uses only Python.

A task can name a link of an articulation as a scene extra, so that RoboLab makes contact sensors for it. RoboLab's
``get_pose`` reads the USD xform of an extra, and the physics does not write that xform: the pose stays at its reset
value while the link moves. The pose of such a link comes from the physics state of its articulation.
"""

from typing import Any


def articulation_links(world: Any) -> dict[str, tuple[str, str]]:
    """Each scene extra of RoboLab's ``world`` whose prim is a link of a scene articulation, mapped to the name of
    that articulation and the name of the link. The prim paths of clone 0 decide the match."""
    links = {
        path: (name, path.rsplit('/', 1)[-1])
        for name, articulation in world.articulations.items()
        for path in articulation.root_physx_view.link_paths[0]
    }
    extras = {name: str(extra.prims[0].GetPath()) for name, extra in world.extras.items()}
    return {name: links[path] for name, path in extras.items() if path in links}


def object_pose(world: Any, name: str, links: dict[str, tuple[str, str]]) -> tuple[Any, Any]:
    """The position in each clone frame and the wxyz rotation in the world frame of the object ``name``, one row per
    clone. ``links`` is ``articulation_links(world)``."""
    if name not in links:
        return world.get_pose(name, is_relative=True)
    pose = world.get_articulation_link_pose(*links[name])
    return pose[:, :3] - world.env.scene.env_origins, pose[:, 3:]
