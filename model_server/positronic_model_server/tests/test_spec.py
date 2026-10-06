"""Descriptions are ordinary JSON data, independent of the component implementation."""

import json

import pytest
from positronic_model_server.spec import ARGS, component, parallel, sequence


# rules-allow: hardcoded-keys — the test pins the existing client's description grammar.
def test_composition_uses_the_client_wire_format():
    spec = sequence(component('scheduler', version=2, fps=20), parallel(component('left'), component('right')))
    assert json.loads(json.dumps(spec)) == {
        'seq': [
            {'name': 'scheduler', 'version': 2, 'args': {'fps': 20}},
            {'par': [{'name': 'left', 'version': 1, 'args': {}}, {'name': 'right', 'version': 1, 'args': {}}]},
        ]
    }


def test_arguments_are_normalized_to_json():
    arguments = {'offsets': (-0.1, 0), 'nested': {'values': [1, None, True]}}
    spec = component('vendor-owned', **arguments)
    assert spec[ARGS] == json.loads(json.dumps(arguments))
    arguments['nested']['values'].append(3)
    assert spec[ARGS]['nested']['values'] == [1, None, True]


@pytest.mark.parametrize('value', [object(), float('inf'), float('nan')])
def test_non_json_arguments_fail_at_construction(value):
    with pytest.raises((TypeError, ValueError)):
        component('vendor-owned', value=value)


@pytest.mark.parametrize('version', [0, -1, True, '1'])
def test_invalid_versions_are_rejected(version):
    with pytest.raises(ValueError, match='positive integer'):
        component('vendor-owned', version=version)


@pytest.mark.parametrize('compose', [sequence, parallel])
def test_empty_composition_is_rejected(compose):
    with pytest.raises(ValueError):
        compose()
