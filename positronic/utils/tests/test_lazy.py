from types import MappingProxyType
from unittest.mock import Mock

import pytest

from positronic.utils.lazy import LazyDict, OverlayMapping


@pytest.mark.parametrize('remove', [(), ('override',)])
def test_overlay_preserves_lazy_values_and_overrides_without_reading_the_source_value(remove):
    kept = Mock(return_value=42)
    shadowed = Mock(side_effect=AssertionError('Overridden value must not be evaluated'))
    replacement = Mock(return_value=7)
    source = LazyDict({}, {'kept': kept, 'override': shadowed})
    overrides = LazyDict({}, {'override': replacement})
    result = OverlayMapping(source, add=overrides, remove=remove)

    assert set(result) == {'kept', 'override'}
    assert len(result) == 2
    assert 'kept' in result and 'override' in result and 'missing' not in result
    kept.assert_not_called()
    shadowed.assert_not_called()
    replacement.assert_not_called()
    assert result['override'] == result['override'] == 7
    replacement.assert_called_once_with()
    assert result['kept'] == 42
    shadowed.assert_not_called()
    with pytest.raises(AssertionError, match='Overridden value must not be evaluated'):
        source['override']


def test_overlay_reflects_updates_to_both_mappings():
    source = {1: 'original', 2: 'kept'}
    overrides = {1: 'replacement'}
    result = OverlayMapping(MappingProxyType(source), add=MappingProxyType(overrides))
    assert dict(result) == {1: 'replacement', 2: 'kept'}
    source[2] = 'updated'
    overrides[3] = 'added'
    del overrides[1]
    assert dict(result) == {1: 'original', 2: 'updated', 3: 'added'}
    assert len(result) == 3
    with pytest.raises(KeyError):
        result[4]


def test_overlay_removes_keys_without_evaluating_values_or_changing_the_source():
    kept = Mock(return_value=42)
    removed = Mock(side_effect=AssertionError('Excluded value must not be evaluated'))
    source = LazyDict({'eager': 1}, {'kept': kept, 'removed': removed})

    result = OverlayMapping[str, int](source, remove=('eager', 'removed', 'missing'))

    assert set(result) == {'kept'}
    assert len(result) == 1
    assert 'kept' in result
    assert 'removed' not in result
    assert result.get('removed') is None
    with pytest.raises(KeyError):
        result['removed']
    assert set(source) == {'eager', 'kept', 'removed'}
    kept.assert_not_called()
    removed.assert_not_called()
    assert result['kept'] == result['kept'] == 42
    kept.assert_called_once_with()
    removed.assert_not_called()


def test_overlay_removes_keys_from_read_only_mappings():
    source = MappingProxyType({'keep': 1, 'remove': 2})
    assert dict(OverlayMapping(source, remove=('remove',))) == {'keep': 1}
    assert source == {'keep': 1, 'remove': 2}


def test_overlay_without_changes_keeps_values_lazy():
    kept = Mock(return_value=42)
    source = LazyDict({}, {'kept': kept})
    result = OverlayMapping[str, int](source)
    assert list(result) == ['kept']
    assert len(result) == 1
    assert 'kept' in result
    kept.assert_not_called()
    assert result['kept'] == 42
    kept.assert_called_once_with()


def test_overlay_reflects_source_changes_and_hides_excluded_keys_added_later():
    source = {1: 'first'}
    result = OverlayMapping(source, remove=(2,))
    source.update({1: 'updated', 2: 'hidden', 3: 'added'})
    assert dict(result) == {1: 'updated', 3: 'added'}
    assert len(result) == 2
    del source[1]
    assert list(result.items()) == [(3, 'added')]


def test_overlay_overrides_take_precedence_over_removal():
    hidden = Mock(side_effect=AssertionError('Excluded value must not be evaluated'))
    source = LazyDict({'kept': 42}, {'hidden': hidden})
    overrides = {}
    result = OverlayMapping(source, add=overrides, remove=('hidden',))
    assert 'hidden' not in result

    overrides['hidden'] = 7
    assert set(result) == {'kept', 'hidden'}
    assert len(result) == 2
    assert 'hidden' in result
    assert result['hidden'] == 7

    del overrides['hidden']
    assert dict(result) == {'kept': 42}
    assert len(result) == 1
    assert 'hidden' not in result
    with pytest.raises(KeyError):
        result['hidden']
    hidden.assert_not_called()


def test_lazy_dict_basic_access():
    """Test that regular dict access works."""
    d = LazyDict({'a': 1, 'b': 2}, {})
    assert d['a'] == 1
    assert d['b'] == 2
    assert d.get('a') == 1
    assert d.get('c', 'default') == 'default'


def test_lazy_dict_lazy_computation():
    """Test that lazy values are computed on first access."""
    call_count = [0]

    def compute_value():
        call_count[0] += 1
        return 42

    d = LazyDict({'a': 1}, {'lazy_key': compute_value})

    # Lazy key should be in keys but not computed yet
    assert 'lazy_key' in d
    assert 'lazy_key' in d.keys()
    assert call_count[0] == 0  # Not computed yet

    # Access triggers computation
    assert d['lazy_key'] == 42
    assert call_count[0] == 1

    # Second access uses cached value
    assert d['lazy_key'] == 42
    assert call_count[0] == 1  # Still 1, not recomputed


def test_lazy_dict_get_triggers_computation():
    """Test that .get() also triggers lazy computation."""
    call_count = [0]

    def compute_value():
        call_count[0] += 1
        return 'computed'

    d = LazyDict({}, {'key': compute_value})

    assert call_count[0] == 0
    assert d.get('key') == 'computed'
    assert call_count[0] == 1


def test_lazy_dict_contains():
    """Test that 'in' operator works for both regular and lazy keys."""
    d = LazyDict({'regular': 1}, {'lazy': lambda: 2})

    assert 'regular' in d
    assert 'lazy' in d
    assert 'missing' not in d


def test_lazy_dict_keys_includes_lazy():
    """Test that keys() includes lazy keys."""
    d = LazyDict({'a': 1}, {'b': lambda: 2, 'c': lambda: 3})

    keys = d.keys()
    assert 'a' in keys
    assert 'b' in keys
    assert 'c' in keys


def test_lazy_dict_copy_preserves_laziness():
    """Test that copy() preserves lazy evaluation."""
    call_count = [0]

    def compute_value():
        call_count[0] += 1
        return 'expensive'

    d = LazyDict({'a': 1}, {'lazy': compute_value})

    # Copy should not trigger computation
    d_copy = d.copy()
    assert call_count[0] == 0

    # Lazy key should still be accessible in copy
    assert 'lazy' in d_copy
    assert call_count[0] == 0  # Still not computed

    # Access in copy triggers computation
    assert d_copy['lazy'] == 'expensive'
    assert call_count[0] == 1


def test_lazy_dict_copy_includes_computed_values():
    """Test that copy includes already-computed lazy values."""
    d = LazyDict({'a': 1}, {'lazy': lambda: 42})

    # Compute the lazy value
    _ = d['lazy']

    # Copy should include the computed value
    d_copy = d.copy()
    assert d_copy['a'] == 1
    assert d_copy['lazy'] == 42


def test_lazy_dict_multiple_lazy_keys():
    """Test multiple lazy keys with independent computation."""
    calls = {'x': 0, 'y': 0}

    def compute_x():
        calls['x'] += 1
        return 'X'

    def compute_y():
        calls['y'] += 1
        return 'Y'

    d = LazyDict({}, {'x': compute_x, 'y': compute_y})

    # Access only x
    assert d['x'] == 'X'
    assert calls['x'] == 1
    assert calls['y'] == 0  # y not computed

    # Access y
    assert d['y'] == 'Y'
    assert calls['y'] == 1


def test_lazy_dict_iter_includes_lazy():
    """Test that iterating includes lazy keys without computing them."""
    call_count = [0]
    d = LazyDict({'a': 1}, {'b': lambda: (call_count.__setitem__(0, call_count[0] + 1), 2)[1]})

    keys = list(d)
    assert set(keys) == {'a', 'b'}
    assert call_count[0] == 0  # Iteration does not trigger computation


def test_lazy_dict_len_includes_lazy():
    """Test that len() counts lazy keys."""
    d = LazyDict({'a': 1}, {'b': lambda: 2, 'c': lambda: 3})
    assert len(d) == 3


def test_lazy_dict_items_includes_lazy():
    """Test that items() includes lazy keys and triggers computation."""
    d = LazyDict({'a': 1}, {'b': lambda: 2})
    result = dict(d.items())
    assert result == {'a': 1, 'b': 2}


def test_lazy_dict_values_includes_lazy():
    """Test that values() includes lazy values and triggers computation."""
    d = LazyDict({}, {'x': lambda: 10})
    assert list(d.values()) == [10]


def test_lazy_dict_dict_conversion():
    """Test that dict(lazy_dict) includes all keys."""
    d = LazyDict({'a': 1}, {'b': lambda: 2})
    plain = dict(d)
    assert plain == {'a': 1, 'b': 2}


def test_lazy_dict_copy_partial_evaluation():
    """Test copy when some lazy keys are evaluated and some are not."""
    calls = {'a': 0, 'b': 0}

    def compute_a():
        calls['a'] += 1
        return 'A'

    def compute_b():
        calls['b'] += 1
        return 'B'

    d = LazyDict({'regular': 1}, {'lazy_a': compute_a, 'lazy_b': compute_b})

    # Evaluate only lazy_a
    _ = d['lazy_a']
    assert calls == {'a': 1, 'b': 0}

    # Copy
    d_copy = d.copy()

    # lazy_a should be in copy as regular value, lazy_b should still be lazy
    assert calls == {'a': 1, 'b': 0}  # No new computation

    # Access lazy_b in copy
    assert d_copy['lazy_b'] == 'B'
    assert calls == {'a': 1, 'b': 1}
