import pytest

from ironic.config import Config


class Env:
    def __init__(self, camera):
        self.camera = camera


class Camera:
    def __init__(self, name: str):
        self.name = name


class MultiEnv:
    def __init__(self, env1: Env, env2: Env):
        self.env1 = env1
        self.env2 = env2


def add(a, b):
    return a + b


static_object = Camera(name="Static Camera")


def test_instantiate_class_object_basic():
    camera_cfg = Config(Camera, name="OpenCV")

    camera_obj = camera_cfg.instantiate()

    assert isinstance(camera_obj, Camera)
    assert camera_obj.name == "OpenCV"


def test_instantiate_class_object_with_function():
    add_cfg = Config(add, a=1, b=2)

    add_obj = add_cfg.instantiate()

    assert add_obj == 3


def test_instantiate_class_object_nested():
    camera_cfg = Config(Camera, name="OpenCV")
    env_cfg = Config(Env, camera=camera_cfg)

    env_obj = env_cfg.instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "OpenCV"


def test_instantiate_class_nested_object_overriden_with_config():
    opencv_camera_cfg = Config(Camera, name="OpenCV")
    luxonis_camera_cfg = Config(Camera, name="Luxonis")

    env_cfg = Config(Env, camera=opencv_camera_cfg)

    env_obj = env_cfg.override({"camera": luxonis_camera_cfg}).instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "Luxonis"


def test_instantiate_class_required_args_provided_with_kwargs_override():
    uncomplete_camera_cfg = Config(Camera)

    camera_obj = uncomplete_camera_cfg.override({"name": "OpenCV"}).instantiate()

    assert isinstance(camera_obj, Camera)
    assert camera_obj.name == "OpenCV"


def test_instantiate_class_required_args_provided_with_path_to_class():
    uncomplete_env_cfg = Config(Env)

    env_obj = uncomplete_env_cfg.override({"camera": "*tests.test_config.static_object"}).instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "Static Camera"


def test_instantiate_set_leaf_value_level2():
    luxonis_camera_cfg = Config(Camera, name="Luxonis")
    env1_cfg = Config(Env, camera=luxonis_camera_cfg)

    env2_cfg = Config(Env)

    multi_env_cfg = Config(MultiEnv, env1=env1_cfg, env2=env2_cfg)

    new_camera_cfg = Config(Camera, name="New Camera")

    full_cfg = multi_env_cfg.override({"env2.camera": new_camera_cfg})
    env_obj = full_cfg.instantiate()

    assert isinstance(env_obj, MultiEnv)
    assert isinstance(env_obj.env1, Env)
    assert isinstance(env_obj.env1.camera, Camera)
    assert env_obj.env1.camera.name == "Luxonis"
    assert isinstance(env_obj.env2, Env)
    assert isinstance(env_obj.env2.camera, Camera)
    assert env_obj.env2.camera.name == "New Camera"
