from pathlib import Path

import positronic.dataset

_DATASET_ROOT = Path(positronic.dataset.__file__).parent


def test_dataset_package_has_no_telemetry_dependency():
    """The dataset core stays agnostic to telemetry: no module under ``positronic/dataset`` imports
    ``positronic.telemetry`` or ``opentelemetry``. Timing rides in as an opaque ``io_span`` context factory
    (default inert), so no telemetry concept leaks into the dataset."""
    offenders = []
    for path in _DATASET_ROOT.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        source = path.read_text()
        if 'telemetry' in source or 'opentelemetry' in source:
            offenders.append(str(path.relative_to(_DATASET_ROOT)))
    assert not offenders, f'dataset modules must not reference telemetry: {offenders}'
