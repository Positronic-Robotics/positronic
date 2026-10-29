from types import MappingProxyType

import numpy as np
import pytest
from positronic_model_server.tree import map_leaves, update_at


def test_map_leaves_preserves_structure_and_does_not_visit_callback_results():
    array = np.arange(3)
    source = MappingProxyType({'nested': [array, (7,)], 'empty': {}})
    visited = []

    def wrap(value):
        visited.append(value)
        return {'wrapped': value}

    result = map_leaves(source, wrap)
    assert len(visited) == 2
    assert visited[0] is array
    assert visited[1] == 7
    assert result['nested'][0]['wrapped'] is array
    assert result['nested'][1] == ({'wrapped': 7},)
    assert result['empty'] == {}
    assert result['nested'] is not source['nested']
    assert source['nested'][0] is array
    assert map_leaves(3, wrap) == {'wrapped': 3}


def test_update_at_copies_only_the_selected_path_and_can_replace_containers():
    untouched = {'keep': []}
    selected = {'old': 1}
    source = MappingProxyType({'literal.key': ([selected], untouched)})
    result = update_at(source, ('literal.key', 0, 0), lambda value: {**value, 'new': 2})
    assert isinstance(result['literal.key'], tuple)
    assert result['literal.key'][0][0] == {'old': 1, 'new': 2}
    assert result['literal.key'][1] is untouched
    assert source['literal.key'][0][0] is selected
    assert selected == {'old': 1}
    assert update_at(source, (), lambda value: len(value)) == 1
    assert update_at([1, 2], (-1,), lambda value: value + 1) == [1, 3]


@pytest.mark.parametrize(
    'source,path,error',
    [({}, ('absent',), KeyError), ([], (0,), IndexError), ([0], ('0',), ValueError), (1, ('a',), ValueError)],
)
def test_update_at_rejects_missing_or_mismatched_paths(source, path, error):
    with pytest.raises(error):
        update_at(source, path, lambda value: value)


@pytest.mark.parametrize('operation', [map_leaves, lambda value, fn: update_at(value, (0,), fn)])
def test_callback_errors_propagate_without_mutating_containers(operation):
    source = [1]

    def fail(value):
        raise RuntimeError('cannot transform')

    with pytest.raises(RuntimeError, match='cannot transform'):
        operation(source, fail)
    assert source == [1]
