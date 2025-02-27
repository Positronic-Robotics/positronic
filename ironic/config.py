import json
import importlib.util
import sys

def custom_encoder(obj):
    if isinstance(obj, Config):
        return obj.to_dict()
    return str(obj)


def _extend_path(path, key):
    if path:
        return path + "." + key
    else:
        return key


def _to_dict(obj):
    if isinstance(obj, Config):
        return obj.to_dict()
    else:
        return obj


def _import_object_from_path(path):
    """
    Import an object from a string path starting with '*'.

    Args:
        path (str): Path to the object in the format "*module.submodule.object"

    Returns:
        The imported object

    Raises:
        ImportError: If the module or object cannot be imported
    """
    if not isinstance(path, str) or not path.startswith('*'):
        return path

    # Remove the leading '*'
    path = path[1:]

    # Split the path to get the module path and object name
    module_parts = path.split('.')
    object_path = module_parts.pop()
    module_path = '.'.join(module_parts)

    # Import the module
    module = importlib.import_module(module_path)

    # Get the object
    return getattr(module, object_path)


class Config:
    def __init__(self, target, *args, **kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def override(self, overrides) -> 'Config':
        cfg = self
        for key, value in overrides.items():
            cfg = cfg._override_single(key, value)
        return cfg

    def _update_flat(self, key, value):
        if isinstance(value, str) and value.startswith('*'):
            value = _import_object_from_path(value)

        if key[0] in '1234567890':
            self.args[int(key)] = value
        else:
            self.kwargs[key] = value

    def _get_value(self, key):
        if key[0] in '1234567890':
            return self.args[int(key)]
        else:
            return self.kwargs[key]


    def _override_single(self, key, value):
        if isinstance(value, str) and value.startswith('*'):
            value = _import_object_from_path(value)

        key_parts = key.split('.')

        current_obj = self

        for part in key_parts[:-1]:
            current_obj = current_obj._get_value(part)

        current_obj._update_flat(key_parts[-1], value)

        return Config(self.target, *self.args, **self.kwargs)

    def instantiate(self):
        # Instantiate any Config objects in args
        instantiated_args = [
            arg.instantiate() if isinstance(arg, Config) else arg
            for arg in self.args
        ]

        # Instantiate any Config objects in kwargs
        instantiated_kwargs = {
            key: value.instantiate() if isinstance(value, Config) else value
            for key, value in self.kwargs.items()
        }

        return self.target(*instantiated_args, **instantiated_kwargs)

    def to_dict(self):
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
        return json.dumps(self.to_dict(), indent=4, default=custom_encoder)
