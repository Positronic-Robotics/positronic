from pkgutil import extend_path as _extend_path

__path__ = _extend_path(__path__, __name__)

from importlib.metadata import version as _version

__version__ = _version('positronic')
