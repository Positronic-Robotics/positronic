from typing import Any

from positronic.policy.base import Policy, SampledPolicy, Session
from positronic.policy.sampler import BalancedSampler, EpisodeCounter, UniformSampler


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

    def new_session(self, context=None, now=None):
        self.session_count += 1
        return _StubSession(self._action, self._meta)

    @property
    def meta(self):
        return self._meta


def _session(key, key_field='ckpt'):
    return _StubSession(meta={key_field: key})


# --- EpisodeCounter ---


def test_counter_records_and_reads_by_key():
    counter = EpisodeCounter('ckpt')
    for _ in range(10):
        counter.record(_session('a'), {})
    assert counter.counts(['a', 'b'], {}) == {'a': 10, 'b': 0}


def test_counter_ignores_session_without_key():
    counter = EpisodeCounter('ckpt')
    counter.record(_StubSession(meta={}), {})
    assert counter.counts(['a'], {}) == {'a': 0}


def test_counter_groups_by_context_fields():
    counter = EpisodeCounter('ckpt', group_fields=('task',))
    for _ in range(5):
        counter.record(_session('a'), {'task': 'task1'})

    # task1 has 5 for 'a'; task2 is a separate, empty group
    assert counter.counts(['a', 'b'], {'task': 'task1'}) == {'a': 5, 'b': 0}
    assert counter.counts(['a', 'b'], {'task': 'task2'}) == {'a': 0, 'b': 0}


def test_counter_no_group_fields_uses_single_global_group():
    counter = EpisodeCounter('ckpt', group_fields=None)
    counter.record(_session('a'), {'task': 'task1'})
    counter.record(_session('a'), {'task': 'task2'})
    counter.record(_session('a'), {'task': 'task3'})
    # All contexts collapse to one global group
    assert counter.counts(['a', 'b'], {'task': 'whatever'}) == {'a': 3, 'b': 0}


def test_counter_counts_only_requested_keys():
    counter = EpisodeCounter('ckpt')
    counter.record(_session('old_checkpoint'), {})
    counter.record(_session('old_checkpoint'), {})
    # Keys not present in the tally read as zero; unknown recorded keys are not surfaced
    assert counter.counts(['new_a', 'new_b'], {}) == {'new_a': 0, 'new_b': 0}


# --- Samplers ---


def test_uniform_sampler_returns_valid_key():
    sampler = UniformSampler()
    keys = ['a', 'b', 'c']
    for _ in range(20):
        assert sampler.sample(keys, {}, {}) in keys


def test_uniform_sampler_with_weights():
    sampler = UniformSampler(weights={'a': 0.0, 'b': 0.0, 'c': 1.0})
    keys = ['a', 'b', 'c']
    for _ in range(20):
        assert sampler.sample(keys, {}, {}) == 'c'


def test_balanced_sampler_favors_underrepresented():
    sampler = BalancedSampler(balance=1)
    keys = ['a', 'b']
    # 10 episodes for 'a', none for 'b'. With balance=1: a=max(10)+1-10=1, b=max(10)+1-0=11
    counts = {'a': 10, 'b': 0}
    picks = {'a': 0, 'b': 0}
    for _ in range(100):
        picks[sampler.sample(keys, {}, counts)] += 1
    assert picks['b'] > picks['a']


def test_balanced_sampler_equal_counts_gives_uniform():
    sampler = BalancedSampler(balance=5)
    keys = ['a', 'b', 'c']
    counts = {'a': 3, 'b': 3, 'c': 3}
    picks = dict.fromkeys(keys, 0)
    for _ in range(300):
        picks[sampler.sample(keys, {}, counts)] += 1
    for k in keys:
        assert 50 < picks[k] < 150


def test_balanced_sampler_all_zero_counts_is_uniform():
    sampler = BalancedSampler(balance=5)
    keys = ['new_a', 'new_b']
    picks = dict.fromkeys(keys, 0)
    for _ in range(100):
        picks[sampler.sample(keys, {}, {'new_a': 0, 'new_b': 0})] += 1
    assert abs(picks['new_a'] - picks['new_b']) < 40


def test_balanced_sampler_reads_counts_through_counter():
    counter = EpisodeCounter('ckpt', group_fields=('task',))
    sampler = BalancedSampler(balance=1)
    keys = ['a', 'b']
    for _ in range(5):
        counter.record(_session('a'), {'task': 'task1'})

    # task1: 'a' is over-represented, so 'b' favored
    picks = {'a': 0, 'b': 0}
    for _ in range(100):
        picks[sampler.sample(keys, {'task': 'task1'}, counter.counts(keys, {'task': 'task1'}))] += 1
    assert picks['b'] > picks['a']

    # task2: untouched group, roughly uniform
    picks = {'a': 0, 'b': 0}
    for _ in range(100):
        picks[sampler.sample(keys, {'task': 'task2'}, counter.counts(keys, {'task': 'task2'}))] += 1
    assert abs(picks['a'] - picks['b']) < 40


# --- SampledPolicy ---


def test_sampled_policy_discovers_keys_from_meta():
    p1 = StubPolicy(meta={'ckpt': '/path/a'})
    p2 = StubPolicy(meta={'ckpt': '/path/b'})
    sampled = SampledPolicy(p1, p2, key_field='ckpt')
    sampled.new_session({})
    assert sampled._keys == ('/path/a', '/path/b')


def test_sampled_policy_delegates_to_sampler():
    p1 = StubPolicy(meta={'ckpt': '/path/a'})
    p2 = StubPolicy(meta={'ckpt': '/path/b'})
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


def test_sampled_policy_session_is_active_sub_policy_session():
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
    assert sampled._keys[0] == '0'
    assert sampled._keys[1] == 'b'


def test_sampled_policy_with_balanced_sampler_uses_its_counter():
    p1 = StubPolicy(meta={'ckpt': 'a'})
    p2 = StubPolicy(meta={'ckpt': 'b'})
    sampled = SampledPolicy(p1, p2, sampler=BalancedSampler(balance=1), key_field='ckpt')

    # 10 completions for 'a', none for 'b' — recorded via the policy's own counter
    for _ in range(10):
        sampled.counter.record(_session('a'), {})

    for _ in range(50):
        sampled.new_session({})
    assert p2.session_count > p1.session_count


def test_sampled_policy_call_delegates():
    p1 = StubPolicy(meta={'ckpt': 'a'}, action={'id': 'a', 'value': 1})
    p2 = StubPolicy(meta={'ckpt': 'b'}, action={'id': 'b', 'value': 2})
    sampled = SampledPolicy(p1, p2, sampler=UniformSampler(weights={'a': 0.0, 'b': 1.0}), key_field='ckpt')
    session = sampled.new_session({})
    assert session({}) == {'id': 'b', 'value': 2}
