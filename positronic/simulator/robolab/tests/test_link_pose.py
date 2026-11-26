"""The ``gt_state`` pose of a RoboLab contact object, read without Isaac Lab from a fake of RoboLab's world state."""

from dataclasses import dataclass, field

import numpy as np

from positronic.simulator.robolab.link_pose import articulation_links, object_pose

_ORIGINS = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
_IDENTITY = [1.0, 0.0, 0.0, 0.0]


@dataclass
class _FakePrim:
    path: str

    def GetPath(self) -> str:  # noqa: N802 -- the USD prim method name
        return self.path


@dataclass
class _FakeXform:
    """A scene extra: a USD xform per clone, whose pose the physics never writes."""

    prims: list[_FakePrim]
    usd_pose: np.ndarray


@dataclass
class _FakeView:
    link_paths: list[list[str]]


@dataclass
class _FakeArticulation:
    root_physx_view: _FakeView
    link_state_w: np.ndarray  # (clones, links, 7): the world pose of each link, as the physics writes it

    @property
    def body_names(self) -> list[str]:
        return [path.rsplit('/', 1)[-1] for path in self.root_physx_view.link_paths[0]]


@dataclass
class _FakeScene:
    env_origins: np.ndarray


@dataclass
class _FakeEnv:
    scene: _FakeScene


@dataclass
class _FakeWorld:
    """The part of RoboLab's ``WorldState`` that the pose read uses, with RoboLab's ``get_pose`` of an extra."""

    extras: dict[str, _FakeXform] = field(default_factory=dict)
    articulations: dict[str, _FakeArticulation] = field(default_factory=dict)
    env: _FakeEnv = field(default_factory=lambda: _FakeEnv(_FakeScene(_ORIGINS)))

    def get_pose(self, name: str, is_relative: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if name in self.extras:
            pose = self.extras[name].usd_pose
        else:
            pose = self.articulations[name].link_state_w[:, 0]
        return pose[:, :3] - self.env.scene.env_origins, pose[:, 3:]

    def get_articulation_link_pose(self, articulation: str, link: str) -> np.ndarray:
        body = self.articulations[articulation]
        return body.link_state_w[:, body.body_names.index(link)]


def _paths(*names: str) -> list[list[str]]:
    return [[f'/World/envs/env_{clone}/scene/{name}' for name in names] for clone in range(len(_ORIGINS))]


def _pose(position: list[float]) -> list[float]:
    return [*position, *_IDENTITY]


def _drawer_world() -> _FakeWorld:
    """A drawer unit whose top drawer the physics moved 0.2 m out in clone 1, as an articulation and two extras."""
    frame = [_pose([x, 0.5, 0.0]) for x, _, _ in _ORIGINS]
    top_at_reset = [_pose([x, 0.5, 0.3]) for x, _, _ in _ORIGINS]
    top_now = [_pose([0.0, 0.5, 0.3]), _pose([4.0, 0.3, 0.3])]
    links = _paths('drawer_unit/frame', 'drawer_unit/top')
    drawer = _FakeArticulation(_FakeView(links), np.array([frame, top_now]).swapaxes(0, 1))
    top = _FakeXform([_FakePrim(row[1]) for row in links], np.array(top_at_reset))
    shelf = _FakeXform([_FakePrim(row[0]) for row in _paths('shelf')], np.array([_pose([0.0, 1.0, 0.0])] * 2))
    return _FakeWorld(extras={'drawer_unit_top': top, 'shelf': shelf}, articulations={'drawer_unit': drawer})


def test_a_link_on_a_joint_has_the_pose_the_physics_writes():
    world = _drawer_world()
    pos, quat = object_pose(world, 'drawer_unit_top', articulation_links(world))
    np.testing.assert_allclose(pos, [[0.0, 0.5, 0.3], [0.0, 0.3, 0.3]])
    np.testing.assert_allclose(quat, [_IDENTITY, _IDENTITY])


def test_an_extra_that_is_no_link_and_an_articulation_root_keep_the_pose_of_robolab():
    world = _drawer_world()
    links = articulation_links(world)
    assert links == {'drawer_unit_top': ('drawer_unit', 'top')}
    for name in ('shelf', 'drawer_unit'):
        pos, quat = object_pose(world, name, links)
        expected_pos, expected_quat = world.get_pose(name)
        np.testing.assert_array_equal(pos, expected_pos)
        np.testing.assert_array_equal(quat, expected_quat)
