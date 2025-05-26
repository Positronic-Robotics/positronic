from collections import deque
import yaml
import importlib.util
from typing import Any, Callable, Dict, Tuple


INSTANTIATE_PREFIX = '@'


class ConfigError(Exception):
    pass


def _to_dict(obj):
    if isinstance(obj, Config):
        return obj._to_dict()
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    else:
        return obj


def _determine_module_by_path(path: str) -> Tuple[str, str]:
    module_path = path.split('.')
    object_path = deque([])

    while len(module_path) > 0:
        try:
            possible_module_path = '.'.join(module_path)
            importlib.import_module(possible_module_path)
            return possible_module_path, '.'.join(object_path)
        except ModuleNotFoundError:
            object_path.appendleft(module_path.pop())

    raise ImportError(f"Module not found for path: {path}")


def _get_object_from_path(module: Any, object_path: str) -> Any:
    x = module
    if object_path:
        for part in object_path.split('.'):
            x = getattr(x, part)
    return x


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

    module_path, object_path = _determine_module_by_path(path)

    # Import the module
    module = importlib.import_module(module_path)
    obj = _get_object_from_path(module, object_path)
    return obj


def _get_base_path_from_default(default: Any) -> str:
    """Extract base path from different types of default values."""
    if isinstance(default, Config):
        return f"{default.target.__module__}.{default.target.__name__}"
    elif isinstance(default, str):
        return default.lstrip(INSTANTIATE_PREFIX)
    elif hasattr(default, "__module__") and hasattr(default, "__name__"):
        return f"{default.__module__}.{default.__name__}"
    else:
        raise ValueError(
            "Default value must be Config, import string or an object with __module__ and __name__"
        )


def _construct_relative_path(value: str, base_path: str) -> str:
    """Construct a new path from a relative import value and base path."""
    parts = base_path.split(".")

    colon_count = len(value) - len(value.lstrip(":"))
    if colon_count > len(parts):
        raise ValueError("Too many ':' prefixes for the default path")

    base_parts = parts[: len(parts) - colon_count]
    remainder = value[colon_count:]

    # Handle different remainder formats:
    # - If remainder starts with ".", it's a relative module path (e.g., ".submodule")
    # - If remainder exists but doesn't start with ".", it's a direct module/object name
    # - If remainder is empty, we use just the base parts
    if remainder.startswith("."):
        new_path = "".join([".".join(base_parts), remainder]) if base_parts else remainder[1:]
    elif remainder:
        new_path = ".".join(base_parts + [remainder]) if base_parts else remainder
    else:
        new_path = ".".join(base_parts)

    return new_path


def _resolve_relative_import(value: str, default: Any) -> Any:
    """Resolve a relative import path (starting with ':')."""
    if default is None:
        raise ValueError("Relative import used with no default value")

    base_path = _get_base_path_from_default(default)
    new_path = _construct_relative_path(value, base_path)
    return _import_object_from_path(f"{INSTANTIATE_PREFIX}{new_path}")


def _resolve_value(value: Any, default: Any | None = None) -> Any:
    """Resolve special strings to actual Python objects.

    Supports two prefixes:

    - ``@`` - absolute import path of the object to instantiate
    - ``:`` - path relative to the current default value

    The ``default`` argument is used when ``value`` starts with ``:``.
    It should either be a :class:`Config` object, a string starting with
    ``@`` or any object that has ``__module__`` and ``__name__``
    attributes.  ``:`` can be repeated multiple times to walk up the
    module hierarchy of the default.
    """
    if isinstance(value, str):
        if value.startswith(INSTANTIATE_PREFIX):
            return _import_object_from_path(value)
        if value.startswith(":"):
            return _resolve_relative_import(value, default)

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
        try:
            default = self._get_value(key)
        except Exception:
            default = None

        value = _resolve_value(value, default)

        if key[0].isdigit():
            self.args[int(key)] = value
        else:
            self.kwargs[key] = value

    def _get_value(self, key):
        if key[0].isdigit():
            return self.args[int(key)]
        else:
            return self.kwargs[key]

    def instantiate(self) -> Any:
        """
        Instatiate the target function with the given arguments and keyword arguments.

        Returns:
            The instantiated target function.
        """
        return self._instantiate_internal()

    def _instantiate_internal(self, path: str = ''):
        """
        Instatiate the target function with the given arguments and keyword arguments.

        Args:
            path (str): The path to the current key. Used for error reporting.

        Returns:
            The instantiated target function.

        Raises:
        """
        def _instantiate_value(value, key, path):
            try:
                if isinstance(value, Config):
                    return value._instantiate_internal(path + f'{key}.')
                elif isinstance(value, (list, tuple)):
                    return type(value)(_instantiate_value(item, f'{key}[{i}]', path) for i, item in enumerate(value))
                elif isinstance(value, dict):
                    return {k: _instantiate_value(v, f'{key}["{k}"]', path) for k, v in value.items()}
                else:
                    return value
            except Exception as e:
                if isinstance(e, ConfigError):
                    raise e
                else:
                    raise ConfigError(f'Error instantiating "{path + key}": {e}') from e

        # Recursively instantiate any Config objects in args
        instantiated_args = [_instantiate_value(arg, key, path) for key, arg in enumerate(self.args)]

        # Recursively instantiate any Config objects in kwargs
        instantiated_kwargs = {
            key: _instantiate_value(value, key, path) for key, value in self.kwargs.items()
        }

        return self.target(*instantiated_args, **instantiated_kwargs)

    def _to_dict(self) -> Dict[str, Any]:
        res = {}

        res["@target"] = f'{INSTANTIATE_PREFIX}{self.target.__module__}.{self.target.__name__}'
        args = [_to_dict(arg) for arg in self.args]
        if len(args) > 0:
            res["*args"] = args
        kwargs = {key: _to_dict(value) for key, value in self.kwargs.items()}
        if len(kwargs) > 0:
            res.update(kwargs)
        return res

    def __str__(self):
        return yaml.dump(self._to_dict(), default_flow_style=False, sort_keys=False)

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
