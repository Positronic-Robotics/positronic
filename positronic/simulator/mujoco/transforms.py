import abc
import contextlib
from pathlib import Path
import pickle
from typing import Any, Dict, Sequence, Tuple

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
    def __init__(self, additional_cameras: Dict[str, Dict[str, Any]]):
        self.additional_cameras = additional_cameras

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        metadata = {s.name: s.data for s in spec.texts}
        model_suffix = metadata.get('model_suffix')

        for camera_name, camera_cfg in self.additional_cameras.items():
            spec.worldbody.add_camera(
                name=f"{camera_name}{model_suffix}",
                pos=camera_cfg['pos'],
                xyaxes=camera_cfg['xyaxes']
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
        assert len(geoms) == 1, f"Expected 1 geom with name {self.object_name}, found {len(geoms)}"
        geoms[0].rgba = np.array(self.color)

        return spec


class AddBox(MujocoSceneTransform):
    def __init__(
            self,
            name: str,
            size: Tuple[float, float, float],
            pos: Tuple[float, float, float] | str,
            quat: Tuple[float, float, float, float] | None = None,
            density: float = 2500,
            rgba: Tuple[float, float, float, float] = (0.5, 0, 0, 1),
            freejoint: bool = False,
    ):
        self.body_name = f"{name}_body"
        self.geom_name = f"{name}_geom"
        self.size = size
        self.pos = pos
        self.quat = quat
        self.density = density
        self.rgba = rgba
        self.freejoint = freejoint

    def apply(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body_dict = {
            'name': self.body_name,
            'pos': self.pos,
        }
        if self.quat is not None:
            body_dict['quat'] = self.quat

        body = spec.worldbody.add_body(**body_dict)

        body.add_geom(
            name=self.geom_name,
            size=self.size,
            density=self.density,
            rgba=self.rgba,
            type=mujoco.mjtGeom.mjGEOM_BOX,
        )

        if self.freejoint:
            body.add_freejoint()

        return spec


class SetBodyPosition(MujocoSceneTransform):
    def __init__(
            self,
            body_name: str,
            *,
            position: Tuple[float, float, float] | None = None,
            quaternion: Tuple[float, float, float, float] | None = None,
            random_position: Tuple[Tuple[float, float, float], Tuple[float, float, float]] | None = None,
            random_euler: Tuple[Tuple[float, float, float], Tuple[float, float, float]] | None = None,
            seed: int | None = None,
    ):
        self.body_name = body_name
        self.seed = seed
        assert (position is None) ^ (random_position is None), "One of position or random_position must be provided"
        assert (quaternion is None) or (random_euler is None), \
            "At most one of quaternion or random_euler must be provided"

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
            assert len(bodies) == 1, f"Expected 1 body with name {self.body_name}, found {len(bodies)}"
            bodies[0].pos = self.position_fn()
            quaternion = self.quaternion_fn()
            if quaternion is not None:
                bodies[0].quat = quaternion
            return spec


def load_model_from_spec_file(
        xml_path: str,
        loaders: Sequence[MujocoSceneTransform] = (),
) -> Tuple[mujoco.MjModel, Dict[str, str]]:
    spec, metadata = load_spec_from_file(xml_path, loaders)
    model = spec.compile()

    return model, metadata


def load_spec_from_file(
        xml_path: str,
        loaders: Sequence[MujocoSceneTransform] = (),
) -> Tuple[mujoco.MjSpec, Dict[str, str]]:
    with open(xml_path, 'r') as f:
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
