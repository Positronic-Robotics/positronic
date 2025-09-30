import os
import pathlib
import pickle
import random
from typing import Any

import fire
import mujoco
import robosuite
import robosuite.models
import robosuite.models.arenas

RANGE_OR_VALUE = float | tuple[float, float]
ASSET_DIR_MESSAGE = """
!!! This spec should be loaded with asset dict.
See simulator.mujoco.transforms.load_model_from_spec !!!
"""
INITIAL_CTRL = [0.3314, -0.4919, -0.5844, -2.0910, -0.2424, 1.7946, 0.6114, 0.0000]


def random_range(value: RANGE_OR_VALUE) -> float:
    if isinstance(value, float):
        return value

    try:
        return random.uniform(value[0], value[1])
    except TypeError as e:
        raise ValueError(f'Invalid range: {value}') from e


def extract_assets(spec: mujoco.MjSpec, parent_dir: str | None = None) -> dict[str, Any]:
    """
    Extract assets from the spec and return a dictionary of asset names to asset data.

    This function is used to create portable mujoco scenes, since original xml files containes
    paths to assets (sometimes relative to meshdir/texturedir and sometimes absolute).

    args:
        spec: mujoco.MjSpec to extract assets from
        parent_dir: directory to prepend to asset paths
    """

    def _load_asset_binary(asset, parent_dir: str) -> bytes:
        asset_path = os.path.join(parent_dir, asset.file)
        with open(asset_path, 'rb') as f:
            data = f.read()
        return data

    assets = {}
    for texture in spec.textures:
        if not texture.file:
            continue

        if not texture.name:
            texture.name = os.path.splitext(os.path.basename(texture.file))[0]

        texture_file_ext = os.path.splitext(texture.file)[1]
        texture_name = f'{texture.name}{texture_file_ext}'
        path = os.path.join(parent_dir, spec.texturedir) if parent_dir else spec.texturedir

        assets[texture_name] = _load_asset_binary(texture, path)
        texture.file = texture_name

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        if not mesh.name:
            mesh.name = os.path.splitext(os.path.basename(mesh.file))[0]

        mesh_file_ext = os.path.splitext(mesh.file)[1]
        mesh_name = f'{mesh.name}{mesh_file_ext}'
        path = os.path.join(parent_dir, spec.meshdir) if parent_dir else spec.meshdir

        assets[mesh_name] = _load_asset_binary(mesh, path)
        mesh.file = mesh_name

    return assets


# TODO: In theory this could be implemented via series of scene transformations. But this is a bit of a pain. :_)
def generate_scene(
    table_height: RANGE_OR_VALUE = 0.3,
    table_size: tuple[float, float, float] = (0.4, 0.6, 0.05),
    portable: bool = True,
) -> mujoco.MjModel:
    table_height = random_range(table_height)
    table_offset = (-0.3, 0, table_height)

    # Create base world with table
    world = robosuite.models.MujocoWorldBase()
    mujoco_arena = robosuite.models.arenas.TableArena(
        table_full_size=table_size,
        table_friction=(1, 0.005, 0.0001),
        table_offset=table_offset,
    )
    mujoco_arena.set_origin([0.8, 0, 0])
    world.merge(mujoco_arena)

    # Add boxes
    asset_dict = {}
    # Load Panda robot specification as base
    panda_spec = mujoco.MjSpec.from_file('assets/mujoco/mjx_panda.xml')
    if portable:
        panda_assets = extract_assets(panda_spec, 'assets/mujoco')
        asset_dict = {**panda_assets}

    # Create temporary spec from the world to merge into Panda spec
    world_spec = mujoco.MjSpec.from_string(world.get_xml())
    if portable:
        world_assets = extract_assets(world_spec, '')
        asset_dict = {**asset_dict, **world_assets}

    # Adjust lighting
    world_spec.lights[0].pos = [1.8, 3.0, 1.5]
    world_spec.lights[0].dir = [-0.2, -0.2, 0]
    world_spec.lights[0].specular = [0.3, 0.3, 0.3]
    world_spec.lights[0].directional = 0
    world_spec.lights[0].castshadow = 0

    # add panda to world
    origin_site = world_spec.worldbody.add_site(name='origin', pos=[0, 0, 0])
    origin_site.attach_body(panda_spec.worldbody, '', '_ph')  # suffix is required by .attach()

    # Configure visual settings
    g = getattr(world_spec.visual, 'global')
    g.azimuth = 120
    g.elevation = -20
    g.offwidth = 1920
    g.offheight = 1080

    if portable:
        world_spec.meshdir = ASSET_DIR_MESSAGE
        world_spec.texturedir = ASSET_DIR_MESSAGE

        world_spec.assets = asset_dict
    else:
        world_spec.meshdir = 'assets/mujoco/assets/'

    # Using custom texts to store metadata about the model
    world_spec.add_text(name='model_suffix', data='_ph')
    world_spec.add_text(name='initial_ctrl', data=','.join([str(x) for x in INITIAL_CTRL]))

    return world_spec, asset_dict


def construct_scene(scene_path: str | pathlib.Path):
    scene_path = pathlib.Path(scene_path)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    assets_path = scene_path.with_suffix('.assets.pkl')
    relative_assets_path = assets_path.name
    spec, assets = generate_scene()

    spec.add_text(name='relative_assets_path', data=str(relative_assets_path))

    spec.compile()

    with open(scene_path, 'w') as f:
        f.write(spec.to_xml())

    with open(assets_path, 'wb') as f:
        pickle.dump(assets, f)


if __name__ == '__main__':
    fire.Fire(construct_scene)
