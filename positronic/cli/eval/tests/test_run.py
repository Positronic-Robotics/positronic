from types import SimpleNamespace
from typing import cast

import pytest

from positronic.cli.eval.run import main
from positronic.eval import Embodiment, Eval, Task


def _eval(simulated: bool) -> Eval:
    return Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=simulated)), task=cast(Task, SimpleNamespace()))


def test_timed_sweep_rejects_real_embodiment(tmp_path):
    """``--timing`` with a real embodiment anywhere in the sweep fails up front: everything under the bound
    tracer enters the report, so a real eval's spans and wall time would silently corrupt it."""
    with pytest.raises(ValueError, match='all-simulated'):
        main(policy=object(), evals=[_eval(True), _eval(False)], output_dir=tmp_path, timing=True)
