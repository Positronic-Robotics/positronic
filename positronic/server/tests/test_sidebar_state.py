"""`loadSidebarState` in the viewer's app.js, driven through node.

The sidebar remembers its width and its open state in localStorage, so every load reads back
whatever an earlier session wrote. A value the browser cannot use must not disable the sidebar.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / 'static' / 'app.js'

# app.js registers listeners when it loads, so the context needs a document. Nothing here runs;
# the harness calls one function and prints what it returns.
_HARNESS = """
const fs = require('fs');
const vm = require('vm');

const stored = JSON.parse(process.argv[1]);
const noop = () => {};
const context = {
  JSON,
  Number,
  Math,
  Object,
  console,
  localStorage: {getItem: () => stored, setItem: noop},
  document: {
    addEventListener: noop,
    querySelector: () => null,
    querySelectorAll: () => [],
    getElementById: () => null,
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
process.stdout.write(JSON.stringify(context.loadSidebarState()));
"""

DEFAULTS = {'isExpanded': False, 'sidebarWidth': 300, 'keyColumnWidth': 150, 'scrollTop': 0}


def load_sidebar_state(stored: str | None) -> dict:
    """Return what app.js reads back from a localStorage holding `stored`."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is required to run the viewer JavaScript')
    result = subprocess.run(
        [node, '-e', _HARNESS, json.dumps(stored), str(APP_JS)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, f'app.js raised on stored state {stored!r}:\n{result.stderr}'
    return json.loads(result.stdout)


# `style.width = "NaNpx"` is invalid CSS, so the browser drops the declaration and the sidebar
# keeps the stylesheet's 0px. A width that is not a number therefore closes the sidebar for good.


@pytest.mark.parametrize(
    'stored',
    [
        pytest.param('not-json{', id='unparseable'),
        pytest.param('', id='empty'),
        pytest.param('{"sidebarWidth": "300px"}', id='width-carries-a-unit'),
        pytest.param('{"sidebarWidth": {}}', id='width-is-an-object'),
        pytest.param('{"sidebarWidth": "wide"}', id='width-is-a-word'),
    ],
)
def test_an_unusable_stored_state_falls_back_to_the_defaults(stored):
    assert load_sidebar_state(stored) == DEFAULTS


@pytest.mark.parametrize(
    'stored',
    [
        pytest.param('not-json{', id='unparseable'),
        pytest.param('{"sidebarWidth": "300px"}', id='width-carries-a-unit'),
        pytest.param('{"sidebarWidth": {}}', id='width-is-an-object'),
    ],
)
def test_an_unusable_stored_width_still_opens_the_sidebar(stored):
    """A width the sidebar opens to must be a real number, or `${width}px` is invalid CSS."""
    width = load_sidebar_state(stored)['sidebarWidth']
    assert isinstance(width, int | float)
    assert width >= 100


# --- the boundary: a usable stored state survives ----------------------------------------------
# The guard above must reject only what the browser cannot use. A state an earlier session
# legitimately wrote is restored unchanged.


def test_a_usable_stored_state_is_restored_unchanged():
    stored = '{"isExpanded": true, "sidebarWidth": 450, "keyColumnWidth": 220, "scrollTop": 90}'
    assert load_sidebar_state(stored) == {
        'isExpanded': True,
        'sidebarWidth': 450,
        'keyColumnWidth': 220,
        'scrollTop': 90,
    }


def test_a_stored_width_below_the_minimum_clamps_rather_than_resets():
    """0 and -500 are numbers the sidebar can use, so they clamp — they do not fall back to 300."""
    assert load_sidebar_state('{"sidebarWidth": 0}')['sidebarWidth'] == 100
    assert load_sidebar_state('{"sidebarWidth": -500}')['sidebarWidth'] == 100


def test_a_numeric_string_width_still_counts_as_a_number():
    assert load_sidebar_state('{"sidebarWidth": "450"}')['sidebarWidth'] == 450


def test_nothing_stored_gives_the_defaults():
    assert load_sidebar_state(None) == DEFAULTS


def test_a_partial_stored_state_keeps_the_defaults_for_what_it_omits():
    assert load_sidebar_state('{"isExpanded": true}') == {**DEFAULTS, 'isExpanded': True}
