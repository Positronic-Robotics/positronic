import abc
import contextlib
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from positronic import geom


@contextlib.contextmanager
def np_seed(seed: int | None = None):
    if seed is None:
        yield
        return

    original_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(original_state)


class MujocoSceneTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        pass


class AddCameras(MujocoSceneTransform):
    def __init__(self, additional_cameras: dict[str, dict[str, Any]]):
        self.additional_cameras = additional_cameras

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        metadata = {s.name: s.data for s in spec.texts}
        model_suffix = metadata.get('model_suffix')

        for camera_name, camera_cfg in self.additional_cameras.items():
            spec.worldbody.add_camera(
                name=f'{camera_name}{model_suffix}', pos=camera_cfg['pos'], xyaxes=camera_cfg['xyaxes']
            )

        return spec


class RecolorObject(MujocoSceneTransform):
    def __init__(self, object_name: str, color: list) -> None:
        super().__init__()
        self.object_name = object_name
        self.color = color

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        # geom with name self.object_name
        geoms = [g for g in spec.geoms if g.name == self.object_name]
        assert len(geoms) == 1, f'Expected 1 geom with name {self.object_name}, found {len(geoms)}'
        geoms[0].rgba = np.array(self.color)

        return spec


class AddBox(MujocoSceneTransform):
    def __init__(
        self,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float] | str,
        quat: tuple[float, float, float, float] | None = None,
        density: float = 2500,
        rgba: tuple[float, float, float, float] = (0.5, 0, 0, 1),
        freejoint: bool = False,
    ):
        self.body_name = f'{name}_body'
        self.geom_name = f'{name}_geom'
        self.size = size
        self.pos = pos
        self.quat = quat
        self.density = density
        self.rgba = rgba
        self.freejoint = freejoint

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body_dict = {'name': self.body_name, 'pos': self.pos}
        if self.quat is not None:
            body_dict['quat'] = self.quat

        body = spec.worldbody.add_body(**body_dict)

        body.add_geom(
            name=self.geom_name, size=self.size, density=self.density, rgba=self.rgba, type=mujoco.mjtGeom.mjGEOM_BOX
        )

        if self.freejoint:
            body.add_freejoint()

        return spec


class SetBodyPosition(MujocoSceneTransform):
    def __init__(
        self,
        body_name: str,
        *,
        position: tuple[float, float, float] | None = None,
        quaternion: tuple[float, float, float, float] | None = None,
        random_position: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        random_euler: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        seed: int | None = None,
    ):
        self.body_name = body_name
        self.seed = seed
        assert (position is None) ^ (random_position is None), 'One of position or random_position must be provided'
        assert (quaternion is None) or (random_euler is None), (
            'At most one of quaternion or random_euler must be provided'
        )

        if position is not None:
            self.position_fn = lambda: position
        else:
            self.position_fn = lambda: np.random.uniform(random_position[0], random_position[1])

        if quaternion is None and random_euler is None:
            self.quaternion_fn = lambda: None
        elif quaternion is not None:
            self.quaternion_fn = lambda: quaternion
        else:

            def quaternion_fn():
                euler = np.random.uniform(random_euler[0], random_euler[1])
                return geom.Rotation.from_euler(euler).as_quat

            self.quaternion_fn = quaternion_fn

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        with np_seed(self.seed):
            bodies = [g for g in spec.bodies if g.name == self.body_name]
            assert len(bodies) == 1, f'Expected 1 body with name {self.body_name}, found {len(bodies)}'
            bodies[0].pos = self.position_fn()
            quaternion = self.quaternion_fn()
            if quaternion is not None:
                bodies[0].quat = quaternion
            return spec


class AddTote(MujocoSceneTransform):
    def __init__(
        self,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float] | str,
        thickness: float = 0.005,
        rgba: tuple[float, float, float, float] = (1, 1, 1, 1),
        density: float = 1000,
    ):
        self.name = name
        self.size = size  # (width, length, height)
        self.pos = pos
        self.thickness = thickness
        self.rgba = rgba
        self.density = density

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.worldbody.add_body(name=f'{self.name}_body', pos=self.pos)
        width, length, height = self.size
        thickness = self.thickness

        # Bottom
        body.add_geom(
            name=f'{self.name}_bottom',
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[width, length, thickness],
            pos=[0, 0, thickness],
            rgba=self.rgba,
            density=self.density,
        )
        # Front
        body.add_geom(
            name=f'{self.name}_front',
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[width, thickness, height],
            pos=[0, -length + thickness, height + thickness],
            rgba=self.rgba,
            density=self.density,
        )
        # Back
        body.add_geom(
            name=f'{self.name}_back',
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[width, thickness, height],
            pos=[0, length - thickness, height + thickness],
            rgba=self.rgba,
            density=self.density,
        )
        # Left
        body.add_geom(
            name=f'{self.name}_left',
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thickness, length - 2 * thickness, height],
            pos=[-width + thickness, 0, height + thickness],
            rgba=self.rgba,
            density=self.density,
        )
        # Right
        body.add_geom(
            name=f'{self.name}_right',
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thickness, length - 2 * thickness, height],
            pos=[width - thickness, 0, height + thickness],
            rgba=self.rgba,
            density=self.density,
        )

        body.add_freejoint()

        return spec


class AddObjectsInTote(MujocoSceneTransform):
    def __init__(
        self,
        tote_name: str,
        object_name_prefix: str,
        num_objects: int,
        object_size: tuple[float, float, float],
        tote_size: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
        seed: int | None = None,
    ):
        self.tote_name = tote_name
        self.object_name_prefix = object_name_prefix
        self.num_objects = num_objects
        self.object_size = object_size
        self.tote_size = tote_size
        self.rgba = rgba
        self.seed = seed

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        tote_body = next(b for b in spec.bodies if b.name == f'{self.tote_name}_body')
        tote_pos = tote_body.pos

        width, length, height = self.tote_size

        # Define placement area inside the tote (accounting for walls)
        margin = 0.02
        min_x = tote_pos[0] - width + margin
        max_x = tote_pos[0] + width - margin
        min_y = tote_pos[1] - length + margin
        max_y = tote_pos[1] + length - margin

        with np_seed(self.seed):
            for i in range(self.num_objects):
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                # tote_pos[2] is the bottom of the tote body.
                # The bottom geom has thickness 0.005 (default) and is at pos=[0, 0, t].
                # So the floor of the tote is at tote_pos[2] + 2*t.
                # We add a small margin + stacking.
                # Assuming default thickness of 0.005 for now as it is not passed in.
                tote_thickness = 0.005
                z = tote_pos[2] + 2 * tote_thickness + self.object_size[2] + i * 2 * self.object_size[2]

                body = spec.worldbody.add_body(name=f'{self.object_name_prefix}_{i}_body', pos=[x, y, z])
                body.add_geom(
                    name=f'{self.object_name_prefix}_{i}_geom',
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=self.object_size,
                    rgba=self.rgba,
                    density=1000,
                )
                body.add_freejoint()

        return spec


class SetTwoObjectsPositions(MujocoSceneTransform):
    def __init__(
        self,
        object1_name: str,
        object2_name: str,
        table_bounds: tuple[tuple[float, float], tuple[float, float]],
        min_distance: float,
        object_sizes: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        seed: int | None = None,
    ):
        self.object1_name = object1_name
        self.object2_name = object2_name
        self.table_bounds = table_bounds  # ((min_x, max_x), (min_y, max_y))
        self.min_distance = min_distance
        self.object_sizes = object_sizes
        self.seed = seed

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        with np_seed(self.seed):
            body1 = next(b for b in spec.bodies if b.name == f'{self.object1_name}_body')
            body2 = next(b for b in spec.bodies if b.name == f'{self.object2_name}_body')

            (min_x, max_x), (min_y, max_y) = self.table_bounds

            for _ in range(100):  # Max attempts
                pos1 = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y), body1.pos[2]]
                pos2 = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y), body2.pos[2]]

                dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

                overlap = False
                if self.object_sizes is not None:
                    s1, s2 = self.object_sizes
                    # Check AABB overlap
                    # sizes are half-sizes (w, l, h)
                    # Overlap if |x1 - x2| < w1 + w2 AND |y1 - y2| < l1 + l2
                    # We add a small margin to min_distance effectively
                    if abs(pos1[0] - pos2[0]) < (s1[0] + s2[0]) and abs(pos1[1] - pos2[1]) < (s1[1] + s2[1]):
                        overlap = True

                if dist >= self.min_distance and not overlap:
                    body1.pos = pos1
                    body2.pos = pos2
                    return spec

            raise RuntimeError('Could not place objects without overlap after 100 attempts')


def load_model_from_spec_file(
    xml_path: str, loaders: Sequence[MujocoSceneTransform] = ()
) -> tuple[mujoco.MjModel, dict[str, str]]:
    spec, metadata = load_spec_from_file(xml_path, loaders)
    model = spec.compile()

    return model, metadata


def load_spec_from_file(
    xml_path: str, loaders: Sequence[MujocoSceneTransform] = ()
) -> tuple[mujoco.MjSpec, dict[str, str]]:
    with open(xml_path) as f:
        xml_string = f.read()

    spec = mujoco.MjSpec.from_string(xml_string)

    metadata = {s.name: s.data for s in spec.texts}

    for loader in loaders:
        spec = loader.apply(spec)

    assets = metadata.get('relative_assets_path')

    if assets is not None:
        assets_path = Path(xml_path).parent / assets
        with open(assets_path, 'rb') as f:
            assets = pickle.load(f)
        spec.assets = assets

    return spec, metadata
