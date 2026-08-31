"""The LIBERO env-server e2e, as a pytest.

It drives the real env-server subprocess, which needs LIBERO's 3.10 interpreter and offscreen rendering: a demo
replay from a tiny committed ``.npz`` fixture, and the task list the benchmark itself answers. That is too heavy
for the normal suite, so this path is kept out of the default ``testpaths``: the dedicated
``.github/workflows/libero-e2e.yaml`` lane runs it by explicit path (Linux software rendering,
``MUJOCO_GL=osmesa``); a plain ``uv run pytest`` never collects it.

The fixture is generated once on a LIBERO box::

    uv run --no-project positronic/simulator/libero/make_fixture.py \
        --demo-path "$LIBERO_DATASETS/libero_spatial/<task>_demo.hdf5" \
        --out positronic/simulator/libero/tests/libero_spatial_task0.npz

then run by explicit path (macOS renders via GLFW, no env var; Linux needs ``MUJOCO_GL=osmesa``)::

    uv run --locked pytest positronic/simulator/libero/tests/test_e2e.py
"""

import os

import pytest

from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.libero.e2e import run_replay
from positronic.simulator.libero.launcher import serve_libero

_FIXTURE = os.path.join(os.path.dirname(__file__), 'libero_spatial_task0.npz')


@pytest.mark.skipif(
    not os.path.exists(_FIXTURE), reason='e2e fixture missing — generate it on a LIBERO box with make_fixture.py'
)
@pytest.mark.timeout(900)  # the env server bootstraps its 3.10 deps on first run, which can take minutes
@pytest.mark.parametrize('command_mode', ['cartesian', 'cartesian_delta'])
def test_demo_replay_reaches_success(command_mode):
    rate = run_replay(_FIXTURE, suite='libero_spatial', task_id=0, command_mode=command_mode)
    assert rate == 1.0, f'demo replay ({command_mode}) rate {rate:.2f} — every prerecorded demo must replay to success'


@pytest.mark.timeout(900)
def test_the_env_lists_the_tasks_a_spec_selects():
    """The eval keeps no task count, so LIBERO's own registry is what a sweep is built from. This lane is the
    only one that reaches it: positronic's interpreter cannot import LIBERO."""
    with serve_libero() as (host, port):
        conn = EnvConnection(host, port)
        try:
            records = conn.tasks({'suite': 'libero_spatial', 'task_id': None})
            assert [r['task_id'] for r in records] == list(range(len(records)))
            assert {r['suite'] for r in records} == {'libero_spatial'}

            pinned = conn.tasks({'suite': 'libero_spatial', 'task_id': 3})
            assert pinned == [{'suite': 'libero_spatial', 'task_id': 3}]

            pair = conn.tasks({'suite': ['libero_spatial', 'libero_object'], 'task_id': None})
            assert {r['suite'] for r in pair} == {'libero_spatial', 'libero_object'}

            # A task nobody has: the server raises and the client re-raises, so a wrong selection stops the run.
            with pytest.raises(RuntimeError):
                conn.tasks({'suite': 'libero_spatial', 'task_id': 999})
        finally:
            conn.close()
