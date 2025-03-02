import pytest

import ironic as ir


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
    camera_cfg = ir.Config(Camera, name="OpenCV")

    camera_obj = camera_cfg.instantiate()

    assert isinstance(camera_obj, Camera)
    assert camera_obj.name == "OpenCV"


def test_instantiate_class_object_with_function_created():
    add_cfg = ir.Config(add, a=1, b=2)

    add_obj = add_cfg.instantiate()

    assert add_obj == 3


def test_instantiate_class_object_nested_created():
    camera_cfg = ir.Config(Camera, name="OpenCV")
    env_cfg = ir.Config(Env, camera=camera_cfg)

    env_obj = env_cfg.instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "OpenCV"


def test_instantiate_class_nested_object_overriden_with_config_created():
    opencv_camera_cfg = ir.Config(Camera, name="OpenCV")
    luxonis_camera_cfg = ir.Config(Camera, name="Luxonis")

    env_cfg = ir.Config(Env, camera=opencv_camera_cfg)

    env_obj = env_cfg.override(camera=luxonis_camera_cfg).instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "Luxonis"


def test_instantiate_class_required_args_provided_with_kwargs_override_created():
    incomplete_camera_cfg = ir.Config(Camera)

    camera_obj = incomplete_camera_cfg.override(name="OpenCV").instantiate()

    assert isinstance(camera_obj, Camera)
    assert camera_obj.name == "OpenCV"


def test_instantiate_class_required_args_provided_with_path_to_class_created():
    incomplete_env_cfg = ir.Config(Env)

    env_obj = incomplete_env_cfg.override(camera="@tests.test_config.static_object").instantiate()

    assert isinstance(env_obj, Env)
    assert isinstance(env_obj.camera, Camera)
    assert env_obj.camera.name == "Static Camera"


def test_instantiate_set_leaf_value_level2_created():
    luxonis_camera_cfg = ir.Config(Camera, name="Luxonis")
    env1_cfg = ir.Config(Env, camera=luxonis_camera_cfg)

    env2_cfg = ir.Config(Env)

    multi_env_cfg = ir.Config(MultiEnv, env1=env1_cfg, env2=env2_cfg)

    new_camera_cfg = ir.Config(Camera, name="New Camera")

    full_cfg = multi_env_cfg.override(env2=ir.Config(Env, camera=new_camera_cfg))
    env_obj = full_cfg.instantiate()

    assert isinstance(env_obj, MultiEnv)
    assert isinstance(env_obj.env1, Env)
    assert isinstance(env_obj.env1.camera, Camera)
    assert env_obj.env1.camera.name == "Luxonis"
    assert isinstance(env_obj.env2, Env)
    assert isinstance(env_obj.env2.camera, Camera)
    assert env_obj.env2.camera.name == "New Camera"


def test_override_basic_keeps_original_config():
    cfg = ir.Config(Camera, name="OpenCV")

    cfg.override(name="New Camera")

    assert cfg.kwargs["name"] == "OpenCV"


def test_override_nested_keeps_original_config():
    cfg = ir.Config(
        MultiEnv,
        env1=ir.Config(
            Env,
            camera=ir.Config(Camera, name="OpenCV")
        ),
        env2=ir.Config(
            Env,
            camera=ir.Config(Camera, name="Luxonis")
        )
    )

    cfg.override(env2=ir.Config(Env, camera=ir.Config(Camera, name="New Camera")))

    assert cfg.kwargs["env2"].kwargs["camera"].kwargs["name"] == "Luxonis"


def test_config_non_callable_target_raises_error():
    # TODO: Another posibility is to return the original object in this case
    non_callable = object()

    with pytest.raises(AssertionError):
        ir.Config(non_callable)


def test_config_to_dict_kwargs_only_produces_correct_dict():
    cfg = ir.Config(Camera, name="OpenCV")

    expected = {
        "@target": f"@{Camera.__module__}.{Camera.__name__}",
        "name": "OpenCV"
    }
    assert cfg.to_dict() == expected


def test_config_to_dict_kwargs_and_args_produces_correct_dict():
    cfg = ir.Config(add, 1, b=2)

    expected = {
        "@target": f"@{add.__module__}.{add.__name__}",
        "*args": [1],
        "b": 2
    }
    assert cfg.to_dict() == expected


def test_config_to_dict_nested_produces_correct_dict():
    cfg = ir.Config(
        MultiEnv,
        env1=ir.Config(Env, camera=ir.Config(Camera, name="OpenCV")),
        env2=ir.Config(Env, camera=ir.Config(Camera, name="Luxonis"))
    )

    expected_dict = {
        "@target": f"@{MultiEnv.__module__}.{MultiEnv.__name__}",
        "env1": {
            "@target": f"@{Env.__module__}.{Env.__name__}",
            "camera": {
                "@target": f"@{Camera.__module__}.{Camera.__name__}",
                "name": "OpenCV"
            }
        },
        "env2": {
            "@target": f"@{Env.__module__}.{Env.__name__}",
            "camera": {
                "@target": f"@{Camera.__module__}.{Camera.__name__}",
                "name": "Luxonis"
            }
        }
    }

    assert cfg.to_dict() == expected_dict


def test_config_str_nested_produces_correct_str():
    cfg = ir.Config(apply, func=ir.Config(add, 1, b=2), a=3, b=4)

    # The exact format that matches the actual output from Config.__str__
    expected_str = f"""'@target': '@{apply.__module__}.{apply.__name__}'
func:
  '@target': '@{add.__module__}.{add.__name__}'
  '*args':
  - 1
  b: 2
a: 3
b: 4
"""

    assert str(cfg) == expected_str


def test_instantiate_not_complete_config_raises_error():
    cfg = ir.Config(Camera)

    with pytest.raises(
        TypeError,
        match="missing 1 required positional argument: 'name'"
    ):
        cfg.instantiate()


def test_config_as_decorator_acts_as_config_class():
    @ir.config
    def sum(a, b):
        return a + b

    assert sum.override(a=1, b=2).instantiate() == 3


def test_config_as_decorator_default_args_are_passed_to_target():
    @ir.config
    def sum(a=1, b=2):
        return a + b

    assert sum.instantiate() == 3


def test_config_as_decorator_override_values_and_instantiate_works():
    @ir.config(a=1, b=2)
    def sum(a, b):
        return a + b

    assert sum.override_and_instantiate() == 3


def test_override_and_instantiate_works_with_flat_configs():
    @ir.config(a=1, b=2)
    def sum(a, b):
        return a + b

    assert sum.override_and_instantiate() == 3


def test_instantiate_with_lists_and_dicts():
    # Define some simple callables for testing
    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    def process_list(items):
        return sum(items)

    def process_dict(data):
        return sum(data.values())

    # Create nested configs in args
    nested_in_list = ir.Config(process_list, [
        ir.Config(add, 1, 2),  # This will be 3
        ir.Config(multiply, 2, 3),  # This will be 6
        5
    ])

    # Create nested configs in kwargs
    nested_in_dict = ir.Config(process_dict, {
        'x': ir.Config(add, 1, 2),  # This will be 3
        'y': ir.Config(multiply, 2, 3),  # This will be 6
        'z': 5
    })

    # Create multi-level nesting
    deeply_nested = ir.Config(
        add,
        ir.Config(multiply, 2, 3),  # This will be 6
        ir.Config(process_dict, {
            'a': ir.Config(add, 1, 2),  # This will be 3
            'b': 4
        })  # This will be 7
    )  # Final result should be 13

    # Test nested configs in lists
    assert nested_in_list.instantiate() == 14  # 3 + 6 + 5 = 14

    # Test nested configs in dictionaries
    assert nested_in_dict.instantiate() == 14  # 3 + 6 + 5 = 14

    # Test deeply nested configs
    assert deeply_nested.instantiate() == 13  # 6 + 7 = 13

    # Test with decorator syntax and nesting
    @ir.config
    def complex_operation(a, items, data):
        return a + sum(items) + sum(data.values())

    result = complex_operation.override(
        a=1,
        items=[ir.Config(add, 1, 2), ir.Config(multiply, 2, 3)],
        data={'x': ir.Config(add, 1, 2), 'y': 5}
    ).instantiate()

    assert result == 18  # 1 + (3 + 6) + (3 + 5) = 18
