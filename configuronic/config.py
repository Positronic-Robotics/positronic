from collections import deque
from types import ModuleType
import yaml
import importlib.util
import inspect
from typing import Any, Callable, Dict, Tuple, List
import posixpath


INSTANTIATE_PREFIX = '@'
RELATIVE_PATH_PREFIX = '.'


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
        return default._creator_module.__name__ + '.' + "stub_name"
    elif isinstance(default, str):
        return default.lstrip(INSTANTIATE_PREFIX)
    elif hasattr(default, "__module__") and hasattr(default, "__name__"):
        return f"{default.__module__}.{default.__name__}"
    elif hasattr(default, "__class__") and hasattr(default, "name"):  # Handle Enum values
        enum_class = default.__class__
        return f"{enum_class.__module__}.{enum_class.__qualname__}.{default.name}"
    else:
        raise ValueError(
            "Default value must be Config, import string, an object with __module__ and __name__, or an Enum value"
        )


def _construct_relative_path(value: str, base_path: str):
    leading = 0
    for leading in range(len(value)):
        if value[leading] != RELATIVE_PATH_PREFIX:
            break

    assert '🤷‍♂️' not in value, "Configuronic does not support relative imports with 🤷‍♂️. Sorry 🤷‍♂️"
    value = '🤷‍♂️' * leading + value[leading:]

    path = base_path + value
    unix_like_path = path.replace('.', '/').replace('🤷‍♂️', '/../')
    unix_like_norm_path = posixpath.normpath(unix_like_path)
    module_path = unix_like_norm_path.replace('/', '.')
    return module_path


def _resolve_relative_import(value: str, default: Any) -> Any:
    """Resolve a relative import path (starting with '.')."""
    if default is None:
        raise ValueError("Relative import used with no default value")

    base_path = _get_base_path_from_default(default)
    new_path = _construct_relative_path(value, base_path)
    return _import_object_from_path(f"{INSTANTIATE_PREFIX}{new_path}")


def _resolve_value(value: Any, default: Any | None = None) -> Any:
    """Resolve special strings to actual Python objects.

    Supports two prefixes:

    - ``@`` - absolute import path of the object to instantiate
    - ``.`` - path relative to the current default value

    The ``default`` argument is used when ``value`` starts with ``.``.
    It should either be a :class:`Config` object, a string starting with
    ``@`` or any object that has ``__module__`` and ``__name__``
    attributes.  ``.`` can be repeated multiple times to walk up the
    module hierarchy of the default.
    """
    if isinstance(value, str):
        if value.startswith(INSTANTIATE_PREFIX):
            if value[len(INSTANTIATE_PREFIX):].startswith(INSTANTIATE_PREFIX):
                return value[len(INSTANTIATE_PREFIX):]
            else:
                return _import_object_from_path(value)
        if value.startswith(RELATIVE_PATH_PREFIX) and not isinstance(default, str):
            return _resolve_relative_import(value, default)

    return value


def _get_value(obj, key):
    if isinstance(obj, Config):
        return obj._get_value(key)
    elif isinstance(obj, list):
        return obj[int(key)]
    elif isinstance(obj, tuple):
        return obj[int(key)]
    elif isinstance(obj, dict):
        return obj[key]
    else:
        raise ConfigError(f"Cannot get value of {obj} with key {key}")


def _set_value(obj, key, value):
    if isinstance(obj, Config):
        obj._set_value(key, value)
    elif isinstance(obj, list):
        obj[int(key)] = value
    elif isinstance(obj, tuple):
        raise NotImplementedError("Overriding tuple values is not implemented")
    elif isinstance(obj, dict):
        obj[key] = value
    else:
        raise ConfigError(f"Cannot set value of {obj} with key {key}")


def _get_creator_module() -> ModuleType:
    current_frame = inspect.currentframe()
    # current frame: this function
    # current frame back: place where this function is called from
    # current frame back back: place one level upperer
    assert current_frame is not None, "Current frame is None. Do your python interpreter support frames?"
    assert current_frame.f_back is not None, "Current frame back is None. Should not happen."
    assert current_frame.f_back.f_back is not None, \
        "Current frame back back is None. This function was probably called from python interpreter."

    module = inspect.getmodule(current_frame.f_back.f_back)

    assert module is not None, "Module is None. Should not happen."

    return module


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
            >>> @cfn.config()
            >>> def sum(a, b):
            >>>     return a + b
            >>> res = sum.override(a=1, b=2).instantiate()
            >>> assert res == 3

            >>> def sum(a, b):
            >>>     return a + b
            >>> res = cfn.Config(sum, a=1, b=2).instantiate()
            >>> assert res == 3
        """
        assert callable(target), f"Target must be callable, got object of type {type(target)}."
        self.target = target
        self.args = list(args)  # TODO: cover argument override with tests
        self.kwargs = kwargs

        self._creator_module = _get_creator_module()

    def override(self, **overrides) -> 'Config':
        """
        Create a new Config with updated parameters.

        Supports nested parameter updates using dot notation (e.g. "model.layers").
        Handles absolute imports (@) and relative imports (.).

        Args:
            **overrides: Parameter paths and their new values.

        Returns:
            Config: A new Config with overridden parameters.

        Raises:
            ConfigError: If parameter path is invalid.

        Example:
            >>> cfg = Config(Pipeline, model=Config(MyModel, layers=6))
            >>> new_cfg = cfg.override(**{"model.layers": 12})
            >>> new_cfg = cfg.override(model="@my_models.CustomModel")
        """
        overriden_cfg = self.copy()
        # we want to keep creator module (module override was called from) for the overriden config
        overriden_cfg._creator_module = _get_creator_module()

        for key, value in overrides.items():
            key_list = key.split('.')

            current_obj = overriden_cfg

            for i, key in enumerate(key_list[:-1]):
                current_obj = _get_value(current_obj, key)
                if current_obj is None:
                    path_to_not_found_arg = '.'.join(key_list[:i + 1])
                    raise ConfigError(f"Argument '{path_to_not_found_arg}' not found in config")

            _set_value(current_obj, key_list[-1], value)

        return overriden_cfg

    def _set_value(self, key, value):
        default = self._get_value(key) if self._has_value(key) else None
        value = _resolve_value(value, default)

        if key[0].isdigit():
            self.args[int(key)] = value
        else:
            self.kwargs[key] = value

    def _get_value(self, key):
        if key[0].isdigit():
            return self.args[int(key)]
        else:
            return self.kwargs.get(key)

    def _has_value(self, key):
        if key[0].isdigit():
            return int(key) < len(self.args)
        else:
            return key in self.kwargs

    def instantiate(self) -> Any:
        """
        Instatiate the target function with the given arguments and keyword arguments.

        Returns:
            The instantiated target function.

        Raises:
            ConfigError: If the target function cannot be instantiated.
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
                    raise ConfigError(f'Error instantiating "{path}{key}": {e}') from e

        # Recursively instantiate any Config objects in args
        instantiated_args = [_instantiate_value(arg, key, path) for key, arg in enumerate(self.args)]

        # Recursively instantiate any Config objects in kwargs
        instantiated_kwargs = {key: _instantiate_value(value, key, path) for key, value in self.kwargs.items()}

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
        return yaml.dump(self._to_dict(), default_flow_style=None, sort_keys=False, width=140)

    def copy(self) -> 'Config':
        """
        Recursively copy config signatures.
        """

        cfg = self._copy()
        cfg._creator_module = _get_creator_module()
        return cfg

    def _copy(self):
        """
        Recursively copy config signatures.
        """

        new_args = [arg._copy() if isinstance(arg, Config) else arg for arg in self.args]

        new_kwargs = {key: value._copy() if isinstance(value, Config) else value for key, value in self.kwargs.items()}

        cfg = Config(self.target, *new_args, **new_kwargs)
        cfg._creator_module = self._creator_module
        return cfg

    def __call__(self, **kwargs):
        """
        Override the config with the given kwargs and instantiate the config.

        Useful for creating a function for a CLI.

        Args:
            **kwargs: Keyword arguments to override the config.

        Returns:
            The instantiated config.

        Example:
            >>> import fire
            >>> @cfn.config()
            >>> def sum(a, b):
            >>>     return a + b
            >>> option1 = sum.override(a=1)
            >>> option2 = sum.override(b=2)
            >>> fire.Fire()
            >>> # Shell call: python script.py option1 --b 5
            >>> # Shell call: python script.py option2 --a 5
        """
        return self.override(**kwargs).instantiate()


def get_required_args(config: Config) -> List[str]:
    """
    Get the list of required arguments to instantiate the target callable.

    Returns:
        List of required argument names (excluding those with default values and those that are already set).
    """
    sig = inspect.signature(config.target)
    required_args = []

    for i, (name, param) in enumerate(sig.parameters.items()):
        if param.default != inspect.Parameter.empty:
            continue
        if param.name in config.kwargs:
            continue
        if i < len(config.args):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # var positional args are not required
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # var keyword args are not required
            continue

        required_args.append(name)
    return required_args


def config(**kwargs) -> Callable[[Callable], Config]:
    """
    Decorator to create a Config object.

    Args:
        **kwargs: Keyword arguments to be passed to the target object.

    Returns:
        A decorator factory that creates a Config object.

    Example:
        >>> @cfn.config(a=1, b=2)
        >>> def sum(a, b):
        >>>     return a + b
        >>> res = sum.instantiate()
        >>> assert res == 3

        >>> @cfn.config()
        >>> def sum(a, b):
        >>>     return a + b
        >>> res = sum.override(a=1, b=2).instantiate()
        >>> assert res == 3
    """
    def _config_decorator(target):
        return Config(target, **kwargs)
    return _config_decorator


def cli(config: Config):
    """
    Run a config object as a CLI.

    Args:
        config: The config object to run.

    Example:
        >>> @cfn.config()
        >>> def sum(a, b):
        >>>     return a + b
        >>> cfn.cli(sum)
        >>> # Shell call: python script.py --a 1 --b 2
        >>> # Shell call: python script.py --help
    """
    import fire

    assert 'help' not in config.kwargs, "Config contains 'help' argument. This is reserved for the help flag."

    def _run_and_help(help: bool = False, **kwargs):
        overriden_config = config.override(**kwargs)
        if help:
            if hasattr(overriden_config.target, '__doc__') and overriden_config.target.__doc__:
                print(overriden_config.target.__doc__)
                print('=' * 140)

            print("Config:")
            for arg in get_required_args(overriden_config):
                print(f"{arg}: <REQUIRED>")
            print()
            print(str(overriden_config))
        else:
            return overriden_config.instantiate()

    fire.Fire(_run_and_help)
