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

    env_obj = incomplete_env_cfg.override(camera="@ironic.tests.test_config.static_object").instantiate()

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
    assert cfg._to_dict() == expected


def test_config_to_dict_kwargs_and_args_produces_correct_dict():
    cfg = ir.Config(add, 1, b=2)

    expected = {
        "@target": f"@{add.__module__}.{add.__name__}",
        "*args": [1],
        "b": 2
    }
    assert cfg._to_dict() == expected


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

    assert cfg._to_dict() == expected_dict


def test_config_str_nested_produces_correct_str():
    cfg = ir.Config(apply, func=ir.Config(add, 1, b=2), a=3, b=4)

    # The exact format that matches the actual output from Config.__str__
    expected_str = f"""'@target': '@{apply.__module__}.{apply.__name__}'
func:
  '@target': '@{add.__module__}.{add.__name__}'
  '*args': [1]
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


def test_instantiate_exception_during_instantiation_has_correct_path():

    def func(a, b):
        return a + b

    def bad_function(a, b):
        raise ValueError("Bad function")

    cfg = ir.Config(func, a=ir.Config(bad_function, a=1, b=2), b=3)

    with pytest.raises(ir.ConfigError) as e:
        cfg.instantiate()

    assert str(e.value) == 'Error instantiating "a": Bad function'


def test_instantiate_exception_during_instantiation_has_correct_path_with_nested_configs():
    def lvl1_function(lvl1arg):
        return lvl1arg

    def lvl2_function(lvl2arg):
        return lvl2arg

    def bad_function(a, b):
        raise ValueError("Bad function")

    cfg = ir.Config(
        lvl1_function,
        lvl1_arg=ir.Config(
            lvl2_function,
            lvl2_arg=ir.Config(bad_function, a=1, b=2),
        ),
    )

    with pytest.raises(ir.ConfigError) as e:
        cfg.instantiate()

    assert str(e.value) == 'Error instantiating "lvl1_arg.lvl2_arg": Bad function'


def test_instantiate_exception_during_instantiation_has_correct_path_with_list():
    def func(list_arg):
        return list_arg[0]

    @ir.config
    def goob_obj():
        return 1

    @ir.config
    def bad_obj():
        raise ValueError("Bad object")

    cfg = ir.Config(func, list_arg=[goob_obj, bad_obj])

    with pytest.raises(ir.ConfigError) as e:
        cfg.instantiate()

    assert str(e.value) == 'Error instantiating "list_arg[1]": Bad object'


def test_instantiate_exception_during_instantiation_has_correct_path_with_dict():
    def func(dict_arg):
        return dict_arg['key']

    @ir.config
    def goob_obj():
        return 1

    @ir.config
    def bad_obj():
        raise ValueError("Bad object")

    cfg = ir.Config(func, dict_arg={'key': goob_obj, 'bad_key': bad_obj})

    with pytest.raises(ir.ConfigError) as e:
        cfg.instantiate()

    assert str(e.value) == 'Error instantiating "dict_arg[\"bad_key\"]": Bad object'


def test_instantiate_override_with_complex_path_to_object_works():
    @ir.config
    def true_if_math_module(obj):
        import math

        return obj is math

    assert true_if_math_module.override(obj="@math").instantiate()


def test_instantiate_override_with_path_to_module_works():
    @ir.config
    def return_true(obj):
        return True

    assert return_true.override(obj="@http.HTTPStatus.OK").instantiate()


def test_override_with_single_colon_relative_path():
    camera_cfg = ir.Config(Camera, name="OpenCV")
    env_cfg = ir.Config(Env, camera=camera_cfg)

    env_obj = env_cfg.override(camera=":static_object").instantiate()

    assert env_obj.camera is static_object


def test_override_with_multiple_colons_relative_path():
    from ironic.tests.support_package.subpkg.a import A
    from ironic.tests.support_package.b import B

    env_cfg = ir.Config(Env, camera=ir.Config(A))

    env_obj = env_cfg.override(camera=":::b.B").instantiate()

    assert env_obj.camera is B


def test_override_with_colon_and_string_default():
    env_cfg = ir.Config(Env, camera="@ironic.tests.test_config.Camera")

    env_obj = env_cfg.override(camera=":static_object").instantiate()

    assert env_obj.camera is static_object


def test_override_with_colon_without_default_raises():
    env_cfg = ir.Config(Env, camera=None)

    with pytest.raises(ValueError):
        env_cfg.override(camera=":static_object")


def test_relative_import_with_enum_default():
    """Test that relative imports work when the default is an Enum value."""
    import http

    def process_enum(enum_val):
        return enum_val.value

    # Set up a config with an enum default
    cfg = ir.Config(process_enum, enum_val=http.HTTPStatus.OK)

    # Override with a relative import (this should work after the fix)
    result_cfg = cfg.override(enum_val=":NOT_FOUND")
    result = result_cfg.instantiate()

    assert result == 404


def test_relative_import_with_nested_enum_default():
    """Test that relative imports work when the default is a nested Enum value.

    Example: geom.Rotation.Representation.ROTVEC
    """
    import http

    def process_nested_enum(enum_val):
        return enum_val.name

    # Set up a config with a nested enum default (HTTPStatus is nested in http module)
    cfg = ir.Config(process_nested_enum, enum_val=http.HTTPStatus.OK)

    # Override with a relative import (this should work after the fix)
    result_cfg = cfg.override(enum_val=":BAD_REQUEST")
    result = result_cfg.instantiate()

    assert result == "BAD_REQUEST"


def test_config_with_list_arg_could_be_overridden():
    @ir.config(a=["a", "b", "c"])
    def join(a):
        return "".join(a)

    assert join.override(**{"a.0": "X"}).instantiate() == "Xbc"


def test_config_with_list_as_arg_preserve_list_type():
    @ir.config(a=[1, 2, 3])
    def return_arg(a):
        return a

    assert isinstance(return_arg.override(**{"a.0": 100}).instantiate(), list)


def test_config_with_list_arg_with_nested_config_could_be_overridden():
    @ir.config(value=1)
    def obj1(value):
        return value

    @ir.config(value=2)
    def obj2(value):
        return value

    @ir.config(arg=[obj1, obj2])
    def return_list(arg):
        return arg

    assert return_list.override(**{"arg.0.value": 100}).instantiate() == [100, 2]


def test_config_with_tuple_arg_with_nested_config_could_be_overridden():
    @ir.config(value=1)
    def obj1(value):
        return value

    @ir.config(value=2)
    def obj2(value):
        return value

    @ir.config(arg=(obj1, obj2))
    def return_tuple(arg):
        return arg

    assert return_tuple.override(**{"arg.0.value": 100}).instantiate() == (100, 2)


def test_config_with_dict_arg_could_be_overridden():
    @ir.config(arg={"a": 1, "b": 2})
    def return_dict(arg):
        return arg

    assert return_dict.override(**{"arg.a": 100}).instantiate() == {"a": 100, "b": 2}


def test_config_with_dict_arg_with_nested_config_could_be_overridden():
    @ir.config(value=1)
    def obj1(value):
        return value

    @ir.config(value=2)
    def obj2(value):
        return value

    @ir.config(arg={"a": obj1, "b": obj2})
    def return_dict(arg):
        return arg

    assert return_dict.override(**{"arg.a.value": 100}).instantiate() == {"a": 100, "b": 2}


def test_required_args_with_no_default_values_returns_all_args():
    @ir.config
    def func(a, b):
        return a + b

    assert ir.get_required_args(func) == ["a", "b"]


def test_required_args_with_default_value_in_function():
    @ir.config
    def func(a, b=1):
        return a + b

    assert ir.get_required_args(func) == ["a"]


def test_required_args_with_default_value_in_config():
    @ir.config(b=1)
    def func(a, b):
        return a + b

    assert ir.get_required_args(func) == ["a"]


def test_required_args_with_default_value_in_config_and_function_returns_all_args():
    @ir.config(a=1)
    def func(a, b=1):
        return a + b

    assert ir.get_required_args(func) == []


def test_required_args_with_args_returns_necessary_args():
    def func(a, b, c):
        return a + b + c

    func = ir.Config(func, 1, 2)

    assert ir.get_required_args(func) == ["c"]


def test_required_args_with_args_and_keyword_only_args_returns_necessary_args():
    def func(a, b, *, c):
        return a + b + c

    func = ir.Config(func, 1, 2)

    assert ir.get_required_args(func) == ["c"]


def test_required_args_with_args_not_required():
    @ir.config
    def func(*args_name):
        return sum(args_name)

    assert ir.get_required_args(func) == []


def test_required_args_with_kwargs_not_required():
    @ir.config
    def func(**kwargs_name):
        return sum(kwargs_name.values())

    assert ir.get_required_args(func) == []


def test_override_existing_list_arg_raises_index_error():
    @ir.config(a=[1, 2, 3])
    def func(a):
        return a

    with pytest.raises(IndexError):
        func.override(**{"a.4": 4})


def test_override_non_existing_list_arg_raises_config_error():
    @ir.config(a=[1, 2, 3])
    def func(a):
        return a

    with pytest.raises(ir.ConfigError, match="Argument 'b' not found in config"):
        func.override(**{"b.0": 4})


def test_override_kwargs_function_with_new_argument_returns_expected_dict():
    @ir.config
    def func(**kwargs):
        return kwargs

    res = func.override(a=4).instantiate()

    assert res == {"a": 4}


def test_override_non_existing_nested_argument_raises_config_error():
    @ir.config(a=1)
    def obj1(a):
        return a

    @ir.config(b=obj1)
    def obj2(b):
        return b

    @ir.config(arg=obj2)
    def composite_obj(arg):
        return arg

    with pytest.raises(ir.ConfigError, match="Argument 'arg.a' not found in config"):
        composite_obj.override(**{"arg.a.b": 4}).instantiate()


if __name__ == "__main__":
    pytest.main()
