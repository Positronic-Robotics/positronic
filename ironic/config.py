import yaml
import importlib.util
from typing import Any, Callable, Dict


INSTANTIATE_PREFIX = '@'


def _to_dict(obj):
    if isinstance(obj, Config):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dict(v) for v in obj]
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
            >>> @ir.Config
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
        overriden_cfg = self.copy()

        for key, value in overrides.items():
            key_list = key.split('.')

            current_obj = overriden_cfg

            for key in key_list[:-1]:
                current_obj = current_obj._get_value(key)

            current_obj._set_value(key_list[-1], value)

        return overriden_cfg

    def _set_value(self, key, value):
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

    def instantiate(self):
        def _instantiate_value(value):
            if isinstance(value, Config):
                return value.instantiate()
            elif isinstance(value, (list, tuple)):
                return type(value)(_instantiate_value(item) for item in value)
            elif isinstance(value, dict):
                return {k: _instantiate_value(v) for k, v in value.items()}
            else:
                return value

        # Recursively instantiate any Config objects in args
        instantiated_args = [_instantiate_value(arg) for arg in self.args]

        # Recursively instantiate any Config objects in kwargs
        instantiated_kwargs = {
            key: _instantiate_value(value) for key, value in self.kwargs.items()
        }

        return self.target(*instantiated_args, **instantiated_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        res = {}

        res["target"] = f'{INSTANTIATE_PREFIX}{self.target.__module__}.{self.target.__name__}'
        args = [_to_dict(arg) for arg in self.args]
        if len(args) > 0:
            res["*args"] = args
        kwargs = {key: _to_dict(value) for key, value in self.kwargs.items()}
        if len(kwargs) > 0:
            res.update(kwargs)
        return res

    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def copy(self) -> 'Config':
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

    def override_and_instantiate(self, **kwargs):
        """
        Override the config with the given kwargs and instantiate the config.

        Useful for creating a function for a CLI.

        Args:
            **kwargs: Keyword arguments to override the config.

        Returns:
            The instantiated config.

        Example:
            >>> import fire
            >>> @ir.config
            >>> def sum(a, b):
            >>>     return a + b
            >>> option1 = sum.override(a=1).override_and_instantiate
            >>> option2 = sum.override(b=2).override_and_instantiate
            >>> fire.Fire()
            >>> # Shell call: python script.py option1 --b 5
            >>> # Shell call: python script.py option2 --a 5
        """
        return self.override(**kwargs).instantiate()


def config(target: Callable | None = None, *args, **kwargs):
    """
    Decorator to create a Config object.

    Args:
        target: The target object to be configured.
        *args: Positional arguments to be passed to the target object.
        **kwargs: Keyword arguments to be passed to the target object.

    Returns:
        The Config object.

    Example:
        >>> @ir.config(a=1, b=2)
        >>> def sum(a, b):
        >>>     return a + b
        >>> res = sum.instantiate()
        >>> assert res == 3

        >>> @ir.config
        >>> def sum(a, b):
        >>>     return a + b
        >>> res = sum.override(a=1, b=2).instantiate()
        >>> assert res == 3
    """

    if target is None:
        def _config_decorator(target):
            return Config(target, *args, **kwargs)
        return _config_decorator
    else:
        return Config(target)
