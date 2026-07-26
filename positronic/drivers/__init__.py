from collections.abc import Iterator
from contextlib import contextmanager
from pkgutil import extend_path as _extend_path

__path__ = _extend_path(__path__, __name__)

# Tail of the error a driver raises when its vendor package is missing. Shared so the drivers cannot drift into
# naming different install paths for the same extra.
HARDWARE_EXTRA_HINT = 'Re-run with the hardware extra:\n  uv run --locked --extra hardware ...\n'


@contextmanager
def vendor_import(package: str, description: str, hint: str = HARDWARE_EXTRA_HINT) -> Iterator[None]:
    """Wrap a driver's vendor import so a missing ``package`` says how to install it.

    Only ``package`` being absent is rewritten. A vendor that is installed but cannot import — a failed
    initialization, or a dependency of its own that is missing — raises with its own name in ``e.name`` and
    propagates untouched, because its diagnostic is the one that says what is actually wrong.
    """
    try:
        yield
    except ModuleNotFoundError as e:
        if e.name != package:
            raise
        raise ModuleNotFoundError(f'{description} is not installed. {hint}') from e
