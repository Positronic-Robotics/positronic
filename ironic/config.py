import yaml
import importlib.util
from typing import Any, Dict

import fire


INSTANTIATE_PREFIX = '@'

def _to_dict(obj):
    if isinstance(obj, Config):
        return obj.to_dict()
    else:
        return obj


def _import_object_from_path(path: str) -> Any:
    """
    Import an object from a string path starting with '@'.

    Args:
        path (str): Path to the object in the format "@module.submodule.object"

    Returns:
        The imported object

    Raises:
        ImportError: If the module or object cannot be imported
    """
    assert path.startswith(INSTANTIATE_PREFIX), f"Path must start with '{INSTANTIATE_PREFIX}'"

    # Remove the leading '@'
    path = path[len(INSTANTIATE_PREFIX):]

    # Split the path to get the module path and object name
    *module_path, object_path = path.split('.')
    module_path = '.'.join(module_path)

    # Import the module
    module = importlib.import_module(module_path)

    # Get the object
    return getattr(module, object_path)


def _resolve_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(INSTANTIATE_PREFIX):
        return _import_object_from_path(value)
    return value


class Config:
    def __init__(self, target, *args, **kwargs):
        """
        Initialize a Config object.

        Stores the callable target and its arguments and keyword arguments, which
        can be overridden/instantiated later.

        Args:
            target: The target object to be configured.
            *args: Positional arguments to be passed to the target object.
            **kwargs: Keyword arguments to be passed to the target object.

        Raises:
            AssertionError: If the target is not callable.

        Example:
            >>> @Config
            >>> def sum(a, b):
            >>>     return a + b
            >>> res = sum.override(a=1, b=2).build()
            >>> assert res == 3
        """
        assert callable(target), f"Target must be callable, got object of type {type(target)}."
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def override(self, **overrides) -> 'Config':
        cfg = self
        for key, value in overrides.items():
            cfg = cfg._override_single(key, value)
        return cfg

    def _update_flat(self, key, value):
        value = _resolve_value(value)

        if key[0].isdigit():
            self.args[int(key)] = value
        else:
            self.kwargs[key] = value

    def _get_value(self, key):
        if key[0].isdigit():
            return self.args[int(key)]
        else:
            return self.kwargs[key]


    def _override_single(self, key, value):
        key_parts = key.split('.')

        overriden_cfg = self.copy()
        current_obj = overriden_cfg

        for part in key_parts[:-1]:
            current_obj = current_obj._get_value(part)

        current_obj._update_flat(key_parts[-1], value)

        return overriden_cfg

    def build(self):
        # Instantiate any Config objects in args
        instantiated_args = [
            arg.build() if isinstance(arg, Config) else arg
            for arg in self.args
        ]

        # Instantiate any Config objects in kwargs
        instantiated_kwargs = {
            key: value.build() if isinstance(value, Config) else value
            for key, value in self.kwargs.items()
        }

        return self.target(*instantiated_args, **instantiated_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        res = {}

        res["target"] = self.target
        args = [_to_dict(arg) for arg in self.args]
        if len(args) > 0:
            res["args"] = args
        kwargs = {key: _to_dict(value) for key, value in self.kwargs.items()}
        if len(kwargs) > 0:
            res["kwargs"] = kwargs
        return res

    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def copy(self):
        """
        Recursively copy config signatures.
        """

        new_args = [
            arg.copy() if isinstance(arg, Config) else arg
            for arg in self.args
        ]

        new_kwargs = {
            key: value.copy() if isinstance(value, Config) else value
            for key, value in self.kwargs.items()
        }

        return Config(self.target, *new_args, **new_kwargs)

    def run_cli(self):
        # TODO: figure out how to use args
        kwargs = fire.Fire(lambda **kwargs: kwargs)

        return self.override(kwargs).build()
