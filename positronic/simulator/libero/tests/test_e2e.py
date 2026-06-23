"""The LIBERO env-server e2e, as a pytest — skipped until a fixture is committed and a LIBERO box runs it.

It replays demo episodes from a tiny committed ``.npz`` fixture through the real env-server subprocess, which
needs LIBERO's 3.10 interpreter and offscreen rendering — so it cannot run in the default suite and skips there
at zero cost. Generate the fixture once on a LIBERO box::

    uv run --no-project positronic/simulator/libero/make_fixture.py \
        --demo-path "$LIBERO_DATASETS/libero_spatial/<task>_demo.hdf5" \
        --out positronic/simulator/libero/tests/libero_spatial_task0.npz

then run it there::

    uv run --locked --extra dev pytest positronic/simulator/libero/tests/test_e2e.py --no-cov
"""

import os

import pytest

from positronic.simulator.libero.e2e import run_replay

_FIXTURE = os.path.join(os.path.dirname(__file__), 'libero_spatial_task0.npz')


@pytest.mark.skipif(
    not os.path.exists(_FIXTURE),
    reason='no e2e fixture committed yet — generate it on a LIBERO box with make_fixture.py',
)
@pytest.mark.timeout(900)  # the env server bootstraps its 3.10 deps on first run, which can take minutes
def test_demo_replay_reaches_success():
    rate = run_replay(_FIXTURE, suite='libero_spatial', task_id=0)
    assert rate == 1.0, f'demo replay success rate {rate:.2f} — every prerecorded demo must replay to success'
