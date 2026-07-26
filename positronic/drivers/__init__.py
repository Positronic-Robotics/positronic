from pkgutil import extend_path as _extend_path

__path__ = _extend_path(__path__, __name__)

# Tail of the ImportError every driver raises when its vendor package is missing. Shared so the drivers cannot
# drift into naming different install paths for the same extra.
HARDWARE_EXTRA_HINT = 'Re-run with the hardware extra:\n  uv run --locked --extra hardware ...\n'
