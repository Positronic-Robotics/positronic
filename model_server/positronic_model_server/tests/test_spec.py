"""Descriptions are ordinary JSON data, independent of the component implementation."""

import json

import pytest
from positronic_model_server.spec import (
    ARGS,
    NAME,
    PAR,
    SEQ,
    VERSION,
    component,
    parallel,
    resolve_params,
    sequence,
    validate,
)


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


@pytest.mark.parametrize(
    'node',
    [
        {},
        {SEQ: []},
        {PAR: []},
        {SEQ: [component('x')], NAME: 'x'},
        {NAME: 'x', VERSION: False},
        {NAME: 'x', ARGS: []},
        {NAME: 'x', ARGS: {'value': float('nan')}},
    ],
)
def test_invalid_description_structure_is_rejected(node):
    with pytest.raises(ValueError):
        validate(node)


def test_structural_validation_does_not_need_a_component_registry():
    validate(sequence(component('unknown-vendor-component', version=17), parallel(component('another'))))


def test_parameter_resolution_refuses_unknown_names_and_copies_defaults():
    defaults = {'options': {'steps': [1, 2]}, 'fps': 20}
    effective = resolve_params(defaults, {'fps': 10})
    effective['options']['steps'].append(3)
    assert defaults['options']['steps'] == [1, 2]
    assert effective['fps'] == 10
    with pytest.raises(ValueError, match='Unknown session parameters'):
        resolve_params(defaults, {'typo': 1})
