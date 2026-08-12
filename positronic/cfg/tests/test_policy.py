"""Tests for the policy configs.

The endpoint field names are spelled out here rather than imported from `policy_cfg`, because these
tests stand in for whatever writes `--policy.endpoints` — which spells them out too, having no way to
import anything. A test written against the constants would follow a rename and pin nothing.

That holds for what a caller WRITES. What a policy REPORTS is read back by first-party code, which
imports, so the meta keys come from the module that produces them.
"""

import configuronic as cfn
import numpy as np
import pos3
import pytest

from positronic import keys
from positronic.cfg import policy as policy_cfg
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.policy.base import SampledPolicy
from positronic.policy.replay import META_DATASET_PATH, META_EPISODE


@pytest.fixture(autouse=True)
def _mirror():
    """A replay endpoint resolves its dataset path through pos3, which needs a live mirror."""
    with pos3.mirror():
        yield


def _dataset(root, episodes: int) -> str:
    """A dataset of ``episodes`` replayable episodes, the nth holding joint waypoints at value n."""
    with LocalDatasetWriter(root) as writer:
        for i in range(episodes):
            with writer.new_episode() as episode:
                for step in range(2):
                    episode.append(
                        keys.TARGET_JOINTS, np.full(7, float(i), dtype=np.float32), ts_ns=10**9 + step * 10**8
                    )
    return str(root)


def _built(**overrides) -> SampledPolicy:
    return policy_cfg.production.override(**overrides).instantiate()


def _session_meta(policy: SampledPolicy) -> dict:
    """The meta of one episode's session — which endpoint it names is decided by the weights."""
    session = policy.new_session({keys.TASK: 'pick'})
    try:
        return dict(session.meta)
    finally:
        session.close()


def test_replay_endpoints_fan_out_and_record_which_recording_ran(tmp_path):
    """The batch a sim rig runs: several endpoints, none of them served."""
    dataset = _dataset(tmp_path / 'ds', episodes=2)
    endpoints = {
        'arm_a': {'kind': 'replay', 'dataset': dataset, 'episode': 0},
        'arm_b': {'kind': 'replay', 'dataset': dataset, 'episode': 1},
    }

    # Weights pin the sampler to one endpoint, which is the only way to ask which policy a label built.
    a = _session_meta(_built(endpoints=endpoints, weights={'arm_a': 1.0, 'arm_b': 0.0}))
    b = _session_meta(_built(endpoints=endpoints, weights={'arm_a': 0.0, 'arm_b': 1.0}))

    assert a[keys.TYPE] == 'replay'
    assert (a[META_DATASET_PATH], a[META_EPISODE]) == (dataset, 0)
    assert (b[META_DATASET_PATH], b[META_EPISODE]) == (dataset, 1)


def test_replay_endpoint_without_an_episode_takes_the_first(tmp_path):
    dataset = _dataset(tmp_path / 'ds', episodes=2)

    meta = _session_meta(_built(endpoints={'arm_a': {'kind': 'replay', 'dataset': dataset}}))

    assert meta[META_EPISODE] == 0


def test_a_served_endpoint_and_a_replay_endpoint_stand_side_by_side(tmp_path):
    """One batch mixing the kinds. Sampling between them dials the served one, so it is not exercised here."""
    dataset = _dataset(tmp_path / 'ds', episodes=1)

    policy = _built(
        endpoints={'served': 'wss://host/api/v1/session', 'recorded': {'kind': 'replay', 'dataset': dataset}}
    )

    assert isinstance(policy, SampledPolicy)


def test_a_bare_endpoint_is_read_as_a_served_url():
    """An entry given as a bare string is dialled, so its URL is what reaches the wire client."""
    with pytest.raises(ValueError, match="Unsupported scheme 'ftp'"):
        _built(endpoints={'served': 'ftp://host/model'})


def test_a_declared_remote_endpoint_dials_its_url():
    with pytest.raises(ValueError, match="Unsupported scheme 'ftp'"):
        _built(endpoints={'served': {'kind': 'remote', 'url': 'ftp://host/model'}})


def test_an_unknown_kind_names_the_kinds_there_are():
    with pytest.raises(ValueError, match="endpoint 'arm_a' declares kind='replayed'; the kinds are remote, replay"):
        _built(endpoints={'arm_a': {'kind': 'replayed', 'dataset': '/tmp/x'}})


def test_a_field_the_kind_does_not_take_is_refused(tmp_path):
    """`dataset_path` is `ReplayPolicy`'s own argument name, and the near-miss a caller reaches for."""
    with pytest.raises(ValueError, match=r"is replay and declares \['dataset_path'\].*takes \['dataset', 'episode'\]"):
        _built(endpoints={'arm_a': {'kind': 'replay', 'dataset_path': '/tmp/x'}})
    with pytest.raises(ValueError, match=r"is remote and declares \['dataset'\]"):
        _built(endpoints={'served': {'kind': 'remote', 'url': 'ws://h:1', 'dataset': '/tmp/x'}})


def test_an_endpoint_that_names_nothing_to_reach_is_refused():
    with pytest.raises(ValueError, match='is replay and names no dataset to play back'):
        _built(endpoints={'arm_a': {'kind': 'replay'}})
    with pytest.raises(ValueError, match='is remote and names no url to dial'):
        _built(endpoints={'served': {'kind': 'remote'}})


def test_an_episode_that_is_not_an_index_is_refused():
    with pytest.raises(ValueError, match="declares episode='2', which is not an episode index"):
        _built(endpoints={'arm_a': {'kind': 'replay', 'dataset': '/tmp/x', 'episode': '2'}})


def test_a_boolean_episode_is_refused():
    """`bool` subclasses `int`, so a JSON `true` reaches the index check as a valid one and plays 1."""
    for episode in (True, False):
        with pytest.raises(ValueError, match=f'declares episode={episode!r}, which is not an episode index'):
            _built(endpoints={'arm_a': {'kind': 'replay', 'dataset': '/tmp/x', 'episode': episode}})


def test_a_relative_dataset_is_refused_in_the_whole_mapping_form():
    """A leading dot is configuronic's relative-import sigil, applied to nested override values too,
    so the override raises rather than carrying the path."""
    for dataset in ('./run', '../data/run', '.run'):
        with pytest.raises(cfn.ConfigError, match="Failed to override 'endpoints'"):
            _built(endpoints={'arm_a': {'kind': 'replay', 'dataset': dataset}})


def test_a_relative_dataset_is_carried_through_the_per_key_form(tmp_path, monkeypatch):
    """The per-key form resolves against the value it replaces, which is a plain string and no base."""
    _dataset(tmp_path / 'ds', episodes=1)
    monkeypatch.chdir(tmp_path)

    meta = _session_meta(
        _built(**{'endpoints': {'arm_a': {'kind': 'replay', 'dataset': 'unset'}}, 'endpoints.arm_a.dataset': './ds'})
    )

    assert meta[META_DATASET_PATH] == './ds'


def test_a_relative_config_reference_inside_an_endpoint_still_resolves():
    """A dotted value nested in an endpoint still resolves to the config it names."""
    built = policy_cfg.production.override(endpoints={'arm_a': {'kind': 'replay', 'dataset': '.replay'}})

    assert isinstance(built.kwargs['endpoints']['arm_a']['dataset'], cfn.Config)


def test_two_replay_endpoints_on_one_recording_are_refused(tmp_path):
    """Both endpoints report one identity, so the set has nothing to tell them apart by."""
    dataset = _dataset(tmp_path / 'ds', episodes=1)
    endpoints = {'arm_a': {'kind': 'replay', 'dataset': dataset}, 'arm_b': {'kind': 'replay', 'dataset': dataset}}

    with pytest.raises(ValueError, match='must be distinguishable'):
        _session_meta(_built(endpoints=endpoints))


def test_reordering_the_endpoints_keeps_each_one_keyed_to_its_own_recording(tmp_path):
    """Reordering the mapping leaves each endpoint's key with the recording that named it."""
    dataset = _dataset(tmp_path / 'ds', episodes=2)
    a = {'kind': 'replay', 'dataset': dataset, 'episode': 0}
    b = {'kind': 'replay', 'dataset': dataset, 'episode': 1}

    forwards = _built(endpoints={'arm_a': a, 'arm_b': b})
    backwards = _built(endpoints={'arm_b': b, 'arm_a': a})

    assert forwards._get_keys() == backwards._get_keys()[::-1]


def test_weights_still_name_endpoints_declared_as_mappings(tmp_path):
    dataset = _dataset(tmp_path / 'ds', episodes=1)

    with pytest.raises(ValueError, match=r"weights name unknown endpoints: \['arm_b'\]"):
        _built(endpoints={'arm_a': {'kind': 'replay', 'dataset': dataset}}, weights={'arm_b': 1.0})
