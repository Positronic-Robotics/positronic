from typing import Any

from positronic.policy.base import Policy, SampledPolicy, Session
from positronic.policy.sampler import BalancedSampler, UniformSampler


class _StubSession(Session):
    def __init__(self, action=None, meta=None):
        self._action = action or []
        self._meta = meta or {}

    def __call__(self, obs):
        return self._action

    @property
    def meta(self):
        return self._meta


class StubPolicy(Policy):
    def __init__(self, meta: dict[str, Any] | None = None, action=None):
        self._meta = meta or {}
        self._action = action
        self.session_count = 0

    def new_session(self, context=None):
        self.session_count += 1
        return _StubSession(self._action, self._meta)

    @property
    def meta(self):
        return self._meta


def test_uniform_sampler_returns_valid_key():
    sampler = UniformSampler()
    keys = ['a', 'b', 'c']
    for _ in range(20):
        assert sampler.sample(keys, {}) in keys


def test_uniform_sampler_with_weights():
    sampler = UniformSampler(weights={'a': 0.0, 'b': 0.0, 'c': 1.0})
    keys = ['a', 'b', 'c']
    for _ in range(20):
        assert sampler.sample(keys, {}) == 'c'


def test_balanced_sampler_favors_underrepresented():
    sampler = BalancedSampler(balance=1)
    keys = ['a', 'b']
    # Record 10 episodes for 'a', none for 'b'
    for _ in range(10):
        sampler.count('a', {})

    # With balance=1, weights are: a=max(10)+1-10=1, b=max(10)+1-0=11
    # 'b' should be sampled much more often
    counts = {'a': 0, 'b': 0}
    for _ in range(100):
        counts[sampler.sample(keys, {})] += 1
    assert counts['b'] > counts['a']


def test_balanced_sampler_groups_by_context():
    sampler = BalancedSampler(balance=1, group_fields=('task',))
    keys = ['a', 'b']

    # Record 5 for 'a' in task1, none for 'b' in task1
    for _ in range(5):
        sampler.count('a', {'task': 'task1'})

    # In task1: a should be disfavored
    counts = {'a': 0, 'b': 0}
    for _ in range(100):
        counts[sampler.sample(keys, {'task': 'task1'})] += 1
    assert counts['b'] > counts['a']

    # In task2: both untouched, should be roughly equal
    counts = {'a': 0, 'b': 0}
    for _ in range(100):
        counts[sampler.sample(keys, {'task': 'task2'})] += 1
    assert abs(counts['a'] - counts['b']) < 40


def test_balanced_sampler_equal_counts_gives_uniform():
    sampler = BalancedSampler(balance=5)
    keys = ['a', 'b', 'c']
    # Equal counts: all weights are max(3)+5-3=5
    for _ in range(3):
        for k in keys:
            sampler.count(k, {})

    counts = dict.fromkeys(keys, 0)
    for _ in range(300):
        counts[sampler.sample(keys, {})] += 1
    # Each should get ~100, allow wide margin
    for k in keys:
        assert 50 < counts[k] < 150


def test_balanced_sampler_ignores_unknown_keys_from_dataset():
    sampler = BalancedSampler(balance=5)
    # Count episodes for keys not in the available policies
    sampler.count('old_checkpoint', {})
    sampler.count('old_checkpoint', {})

    # Available keys are different — should sample uniformly (all zero counts)
    keys = ['new_a', 'new_b']
    counts = dict.fromkeys(keys, 0)
    for _ in range(100):
        counts[sampler.sample(keys, {})] += 1
    assert abs(counts['new_a'] - counts['new_b']) < 40


def test_balanced_sampler_no_group_fields_uses_global():
    sampler = BalancedSampler(balance=1, group_fields=None)
    keys = ['a', 'b']

    # Count with different contexts — all go to same global group
    sampler.count('a', {'task': 'task1'})
    sampler.count('a', {'task': 'task2'})
    sampler.count('a', {'task': 'task3'})

    # 'b' should be favored regardless of context
    counts = {'a': 0, 'b': 0}
    for _ in range(100):
        counts[sampler.sample(keys, {'task': 'whatever'})] += 1
    assert counts['b'] > counts['a']


def test_sampled_policy_discovers_keys_from_meta():
    p1 = StubPolicy(meta={'ckpt': '/path/a'})
    p2 = StubPolicy(meta={'ckpt': '/path/b'})
    sampled = SampledPolicy(p1, p2, key_field='ckpt')
    sampled.new_session({})
    assert sampled._keys == ('/path/a', '/path/b')


def test_sampled_policy_delegates_to_sampler():
    p1 = StubPolicy(meta={'ckpt': '/path/a'})
    p2 = StubPolicy(meta={'ckpt': '/path/b'})
    # Weight=0 for p1, weight=1 for p2 → always picks p2
    sampler = UniformSampler(weights={'/path/a': 0.0, '/path/b': 1.0})
    sampled = SampledPolicy(p1, p2, sampler=sampler, key_field='ckpt')
    for _ in range(10):
        sampled.new_session({})
    assert p1.session_count == 0
    assert p2.session_count == 10


def test_sampled_policy_backward_compat_weights_only():
    p1 = StubPolicy(meta={'ckpt': 'a'})
    p2 = StubPolicy(meta={'ckpt': 'b'})
    sampled = SampledPolicy(p1, p2, weights=[0.0, 1.0], key_field='ckpt')
    for _ in range(10):
        sampled.new_session({})
    assert p1.session_count == 0
    assert p2.session_count == 10


def test_sampled_policy_session_meta_returns_active_sub_policy():
    p1 = StubPolicy(meta={'ckpt': 'a', 'type': 'act'})
    p2 = StubPolicy(meta={'ckpt': 'b', 'type': 'groot'})
    sampled = SampledPolicy(p1, p2, sampler=UniformSampler(weights={'a': 0.0, 'b': 1.0}), key_field='ckpt')
    session = sampled.new_session({})
    assert session.meta == {'ckpt': 'b', 'type': 'groot'}


def test_sampled_policy_fallback_key_when_meta_missing():
    p1 = StubPolicy(meta={})
    p2 = StubPolicy(meta={'ckpt': 'b'})
    sampled = SampledPolicy(p1, p2, key_field='ckpt')
    sampled.new_session({})
    # p1 has no 'ckpt' in meta → falls back to str(index) = '0'
    assert sampled._keys[0] == '0'
    assert sampled._keys[1] == 'b'


def test_sampled_policy_with_balanced_sampler():
    p1 = StubPolicy(meta={'ckpt': 'a'})
    p2 = StubPolicy(meta={'ckpt': 'b'})
    sampler = BalancedSampler(balance=1)
    sampled = SampledPolicy(p1, p2, sampler=sampler, key_field='ckpt')

    # Record 10 completions for 'a', none for 'b'
    for _ in range(10):
        sampler.count('a', {})

    # Now sample many times — 'b' should be strongly favored
    p1.session_count = 0
    p2.session_count = 0
    for _ in range(50):
        sampled.new_session({})
    assert p2.session_count > p1.session_count


def test_sampled_policy_select_action_delegates():
    p1 = StubPolicy(meta={'ckpt': 'a'}, action={'id': 'a', 'value': 1})
    p2 = StubPolicy(meta={'ckpt': 'b'}, action={'id': 'b', 'value': 2})
    sampled = SampledPolicy(p1, p2, sampler=UniformSampler(weights={'a': 0.0, 'b': 1.0}), key_field='ckpt')
    session = sampled.new_session({})
    assert session({}) == {'id': 'b', 'value': 2}
