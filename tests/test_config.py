import pytest

from ironic import Config, config


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


def apply(func, a, b):
    return func(a, b)


static_object = Camera(name="Static Camera")


def test_instantiate_class_object_basic_created():
    camera_cfg = Config(Camera, name="OpenCV")

    camera_obj = camera_cfg.instantiate()

    assert isinstance(camera_obj, Camera)
    assert camera_obj.name == "OpenCV"


def test_instantiate_class_object_with_function_created():
    add_cfg = Config(add, a=1, b=2)

    add_obj = add_cfg.instantiate()

    assert add_obj == 3


def test_instantiate_class_object_nested_created():
    camera_cfg = Config(Camera, name="OpenCV")
    env_cfg = Config(Env, camera=camera_cfg)

    env_obj = env_cfg.instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "OpenCV"


def test_instantiate_class_nested_object_overriden_with_config_created():
    opencv_camera_cfg = Config(Camera, name="OpenCV")
    luxonis_camera_cfg = Config(Camera, name="Luxonis")

    env_cfg = Config(Env, camera=opencv_camera_cfg)

    env_obj = env_cfg.override(camera=luxonis_camera_cfg).instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "Luxonis"


def test_instantiate_class_required_args_provided_with_kwargs_override_created():
    incomplete_camera_cfg = Config(Camera)

    camera_obj = incomplete_camera_cfg.override(name="OpenCV").instantiate()

    assert isinstance(camera_obj, Camera)
    assert camera_obj.name == "OpenCV"


def test_instantiate_class_required_args_provided_with_path_to_class_created():
    incomplete_env_cfg = Config(Env)

    env_obj = incomplete_env_cfg.override(camera="@tests.test_config.static_object").instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "Static Camera"


def test_instantiate_set_leaf_value_level2_created():
    luxonis_camera_cfg = Config(Camera, name="Luxonis")
    env1_cfg = Config(Env, camera=luxonis_camera_cfg)

    env2_cfg = Config(Env)

    multi_env_cfg = Config(MultiEnv, env1=env1_cfg, env2=env2_cfg)

    new_camera_cfg = Config(Camera, name="New Camera")

    full_cfg = multi_env_cfg.override(env2=Config(Env, camera=new_camera_cfg))
    env_obj = full_cfg.instantiate()

    assert isinstance(env_obj, MultiEnv)
    assert isinstance(env_obj.env1, Env)
    assert isinstance(env_obj.env1.camera, Camera)
    assert env_obj.env1.camera.name == "Luxonis"
    assert isinstance(env_obj.env2, Env)
    assert isinstance(env_obj.env2.camera, Camera)
    assert env_obj.env2.camera.name == "New Camera"


def test_override_basic_keeps_original_config():
    cfg = Config(Camera, name="OpenCV")

    cfg.override(name="New Camera")

    assert cfg.kwargs["name"] == "OpenCV"


def test_override_nested_keeps_original_config():
    cfg = Config(
        MultiEnv,
        env1=Config(
            Env,
            camera=Config(Camera, name="OpenCV")
        ),
        env2=Config(
            Env,
            camera=Config(Camera, name="Luxonis")
        )
    )

    cfg.override(env2=Config(Env, camera=Config(Camera, name="New Camera")))

    assert cfg.kwargs["env2"].kwargs["camera"].kwargs["name"] == "Luxonis"


def test_config_non_callable_target_raises_error():
    # TODO: Another posibility is to return the original object in this case
    non_callable = object()

    with pytest.raises(AssertionError):
        Config(non_callable)


def test_config_to_dict_kwargs_only_produces_correct_dict():
    cfg = Config(Camera, name="OpenCV")

    assert cfg.to_dict() == {"target": Camera, "kwargs": {"name": "OpenCV"}}


def test_config_to_dict_kwargs_and_args_produces_correct_dict():
    # TODO: Maybe it will be better to convert args to kwargs?
    cfg = Config(add, 1, b=2)

    assert cfg.to_dict() == {"target": add, "args": [1], "kwargs": {"b": 2}}


def test_config_to_dict_nested_produces_correct_dict():
    cfg = Config(
        MultiEnv,
        env1=Config(Env, camera=Config(Camera, name="OpenCV")),
        env2=Config(Env, camera=Config(Camera, name="Luxonis"))
    )

    expected_dict = {
        "target": MultiEnv,
        "kwargs": {
            "env1": {
                "target": Env,
                "kwargs": {
                    "camera": {
                        "target": Camera,
                        "kwargs": {"name": "OpenCV"}
                    }
                }
            },
            "env2": {
                "target": Env,
                "kwargs": {
                    "camera": {
                        "target": Camera,
                        "kwargs": {"name": "Luxonis"}
                    }
                }
            }
        }
    }

    assert cfg.to_dict() == expected_dict


def test_config_str_nested_produces_correct_str():
    cfg = Config(apply, func=Config(add, 1, b=2), a=3, b=4)

    expected_str = """kwargs:
  a: 3
  b: 4
  func:
    args:
    - 1
    kwargs:
      b: 2
    target: !!python/name:tests.test_config.add ''
target: !!python/name:tests.test_config.apply ''
"""

    assert str(cfg) == expected_str


def test_instantiate_not_complete_config_raises_error():
    cfg = Config(Camera)

    with pytest.raises(
        TypeError,
        match="missing 1 required positional argument: 'name'"
    ):
        cfg.instantiate()


def test_config_as_decorator_acts_as_config_class():
    @config
    def sum(a, b):
        return a + b

    assert sum.override(a=1, b=2).instantiate() == 3


def test_config_as_decorator_default_args_are_passed_to_target():
    @config
    def sum(a=1, b=2):
        return a + b

    assert sum.instantiate() == 3


def test_config_as_decorator_override_values_and_instantiate_works():
    @config(a=1, b=2)
    def sum(a, b):
        return a + b

    assert sum.override_and_instantiate() == 3


def test_override_and_instantiate_works_with_flat_configs():
    @config(a=1, b=2)
    def sum(a, b):
        return a + b

    assert sum.override_and_instantiate() == 3
