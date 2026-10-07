"""The i2rt YAM arm: the driver in `driver`, its settle tuning in `settle`."""

import importlib

_SUBMODULES = ('driver', 'settle')


def __getattr__(name: str):
    # The driver loads on first use: it needs the yam extra, and the configs import `settle` without it.
    if name in _SUBMODULES or name.startswith('__'):
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module(f'{__name__}.driver'), name)
