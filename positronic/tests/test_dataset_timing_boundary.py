from pathlib import Path

import positronic.dataset

_DATASET_ROOT = Path(positronic.dataset.__file__).parent


def test_dataset_package_has_no_eval_timing_dependency():
    """The dataset core stays agnostic to telemetry (comment 4): no module under ``positronic/dataset``
    imports ``positronic.eval_timing`` or hard-codes the producer's ``timing.`` name prefix. Timing rides in
    as opaque ``(name, value)`` pairs through ``TimingHooks``; the writer never learns what they mean."""
    imports_eval_timing = []
    hardcodes_prefix = []
    for path in _DATASET_ROOT.rglob('*.py'):
        src = path.read_text()
        if 'eval_timing' in src:
            imports_eval_timing.append(str(path.relative_to(_DATASET_ROOT)))
        if "'timing.'" in src:
            hardcodes_prefix.append(str(path.relative_to(_DATASET_ROOT)))
    assert not imports_eval_timing, f'dataset modules must not reference eval_timing: {imports_eval_timing}'
    assert not hardcodes_prefix, f"dataset modules must not hard-code the 'timing.' prefix: {hardcodes_prefix}"
