import operator
import pickle

import numpy as np
import pytest

from pimm import Time


class TestTime:
    def test_snapshot_and_named_access(self):
        source = {'world': 100, 'tick': 2}
        timestamps = Time(**source)
        source['world'] = 200
        source['wall'] = 300
        assert timestamps.timelines == ('world', 'tick')
        assert timestamps['world'] == 100
        assert dict(timestamps) == {'world': 100, 'tick': 2}
        assert list(timestamps.values()) == [100, 2]
        assert 'world' in timestamps
        assert ('world', 'tick') not in timestamps

    @pytest.mark.parametrize('mutation, args', [(operator.setitem, ('world', 300)), (operator.delitem, ('world',))])
    def test_read_only_access(self, mutation, args):
        timestamps = Time(world=100)
        with pytest.raises(TypeError):
            mutation(timestamps, *args)

    @pytest.mark.parametrize('coordinates', [{}, {'': 1}, {'  ': 1}])
    def test_invalid_names(self, coordinates):
        with pytest.raises(ValueError):
            Time(**coordinates)

    def test_dynamic_names_include_constructor_argument_names(self):
        coordinates = {'server.wall': 100, 'server/wall': 200, 'self': 300, 'timestamps': 400}
        assert dict(Time(**coordinates)) == coordinates

    @pytest.mark.parametrize('coordinates', [{1: 2}, {None: 2}])
    def test_names_must_be_strings(self, coordinates):
        with pytest.raises(TypeError, match='keywords must be strings'):
            Time(**coordinates)

    @pytest.mark.parametrize('value', [1.5, 1.0, '1', None, True, False, np.bool_(True), np.bool_(False)])
    def test_coordinates_must_be_integers(self, value):
        with pytest.raises(TypeError):
            Time(world=value)

    def test_numpy_integers_and_negative_coordinates(self):
        timestamps = Time(world=np.int64(-100), tick=np.uint64(2))
        assert timestamps == {'world': -100, 'tick': 2}
        assert type(timestamps['world']) is int

    def test_selection_preserves_requested_order(self):
        timestamps = Time(world=100, wall=1000, tick=2)
        selected = timestamps[('tick', 'world')]
        assert isinstance(selected, Time)
        assert selected.timelines == ('tick', 'world')
        assert selected == Time(world=100, tick=2)
        assert timestamps[('world',)] == Time(world=100)

    @pytest.mark.parametrize('names', [(), ('world', 'world')])
    def test_invalid_selection(self, names):
        with pytest.raises(ValueError):
            Time(world=100)[names]

    @pytest.mark.parametrize('names', ['missing', ('world', 'missing')])
    def test_missing_timeline(self, names):
        with pytest.raises(KeyError, match='missing'):
            Time(world=100)[names]

    @pytest.mark.parametrize(
        'right, equal, before, after',
        [
            ({'tick': 2, 'world': 100}, True, False, False),
            ({'tick': 3, 'world': 100}, False, True, False),
            ({'tick': 2, 'world': 200}, False, True, False),
            ({'tick': 1, 'world': 50}, False, False, True),
            ({'tick': 1, 'world': 200}, False, False, False),
        ],
    )
    def test_componentwise_order(self, right, equal, before, after):
        left = Time(world=100, tick=2)
        right = Time(**right)
        assert (left == right) is equal
        assert (left != right) is not equal
        assert (left < right) is before
        assert (left <= right) is (before or equal)
        assert (left > right) is after
        assert (left >= right) is (after or equal)
        assert (right > left) is before
        assert (right >= left) is (before or equal)

    def test_order_is_transitive(self):
        first = Time(world=100, tick=2)
        second = Time(tick=3, world=100)
        third = Time(world=200, tick=3)
        assert first < second < third
        assert first < third

    @pytest.mark.parametrize(
        'operation', [operator.lt, operator.le, operator.gt, operator.ge, operator.add, operator.sub]
    )
    @pytest.mark.parametrize('coordinates', [{'world': 100}, {'world': 100, 'tick': 2, 'wall': 300}, {'other': 100}])
    def test_operations_require_same_names(self, operation, coordinates):
        left = Time(world=100, tick=2)
        right = Time(**coordinates)
        assert left != right
        with pytest.raises(ValueError, match='same timeline names'):
            operation(left, right)

    @pytest.mark.parametrize(
        'operation', [operator.lt, operator.le, operator.gt, operator.ge, operator.add, operator.sub]
    )
    @pytest.mark.parametrize('other', [1, {'world': 100}])
    def test_operations_require_timestamp_values(self, operation, other):
        timestamps = Time(world=100)
        with pytest.raises(TypeError):
            operation(timestamps, other)
        with pytest.raises(TypeError):
            operation(other, timestamps)

    def test_arithmetic_matches_names_and_retains_operands(self):
        left = Time(world=100, tick=2)
        right = Time(tick=3, world=20)
        assert left + right == Time(world=120, tick=5)
        assert left - right == Time(world=80, tick=-1)
        assert right - left == Time(tick=1, world=-80)
        assert (left + right) - right == left
        assert left == {'world': 100, 'tick': 2}
        assert right == {'tick': 3, 'world': 20}


def test_time_pickle_preserves_coordinates_and_immutability():
    time = Time(**{'camera.capture': 42, 'emitted.wall': 100})
    restored = pickle.loads(pickle.dumps(time))
    assert restored == time
    with pytest.raises(TypeError):
        operator.setitem(restored, 'camera.capture', 99)
