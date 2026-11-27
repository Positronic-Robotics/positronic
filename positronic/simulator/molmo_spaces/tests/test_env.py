"""The MolmoSpaces behavior that the server changes, checked in the MolmoSpaces interpreter.

Importing the server needs the asset packs, so the tests skip when ``MLSPACES_ASSETS_DIR`` is unset.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from positronic.simulator.molmo_spaces import launcher, mapping

# The probe runs in the MolmoSpaces interpreter, where positronic does not import; it names its bodies as below.
_PROBE = Path(__file__).with_name('receptacle_probe.py')
_RECEPTACLE, _PICKUP, _WORLD = 'receptacle', 'pickup', 'world'


@pytest.fixture(scope='module')
def receptacle_scenes() -> dict[str, dict[str, Any]]:
    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        pytest.skip(f'{mapping.ASSETS_DIR_ENV} is unset — importing the server needs the MolmoSpaces asset packs')
    result = subprocess.run(
        [str(launcher.ensure_molmo_venv()), str(_PROBE)],
        env=launcher.molmo_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_a_receptacle_contact_with_a_body_that_is_not_checked_computes_no_box_of_that_body(receptacle_scenes):
    scene = receptacle_scenes['on_floor']
    assert scene['contacts'] > 0
    assert scene['found'] == []
    assert _WORLD not in scene['boxed']


def test_the_fallback_computes_one_box_for_a_receptacle_of_many_geoms(receptacle_scenes):
    # rules-allow: implementation-bound-tests — the box count is the cost that the server bounds.
    assert receptacle_scenes['on_floor']['boxed'].count(_RECEPTACLE) == 1


def test_an_object_that_touches_the_receptacle_from_above_is_on_it(receptacle_scenes):
    scene = receptacle_scenes['held_by_contact']
    assert scene['found'] == [_PICKUP]
    assert _WORLD not in scene['boxed']


def test_an_object_just_above_the_receptacle_is_on_it(receptacle_scenes):
    scene = receptacle_scenes['just_above']
    assert scene['found'] == [_PICKUP]
    assert _WORLD not in scene['boxed']
