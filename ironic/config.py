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
    if not path.startswith('*'):
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

    def instantiate(self, overrides=None, __path: str = ""):
        if overrides is None:
            overrides = {}

        # Create a copy of overrides to avoid modifying the original
        overrides_copy = overrides.copy()

        # Process any string paths in overrides
        for key, value in overrides_copy.items():
            if isinstance(value, str) and value.startswith('*'):
                overrides_copy[key] = _import_object_from_path(value)

        args = []
        for i, arg in enumerate(self.args):
            next_path = _extend_path(__path, str(i))
            if next_path in overrides_copy:
                arg = overrides_copy.pop(next_path)
                # Process string paths in args
                if isinstance(arg, str) and arg.startswith('*'):
                    arg = _import_object_from_path(arg)
            if isinstance(arg, Config):
                arg = arg.instantiate(overrides_copy, next_path)
            args.append(arg)

        kwargs = {}
        for key, value in self.kwargs.items():
            next_path = _extend_path(__path, key)
            if next_path in overrides_copy:
                value = overrides_copy.pop(next_path)
                # Process string paths in kwargs
                if isinstance(value, str) and value.startswith('*'):
                    value = _import_object_from_path(value)
            if isinstance(value, Config):
                value = value.instantiate(overrides_copy, next_path)
            kwargs[key] = value

        for key, value in list(overrides_copy.items()):
            if __path == "" and "." not in key:
                # Process string paths in top-level overrides
                if isinstance(value, str) and value.startswith('*'):
                    value = _import_object_from_path(value)
                kwargs[key] = value
                overrides_copy.pop(key)

        return self.target(*args, **kwargs)

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
