"""Tests for the policy configs.

The endpoint field names are spelled out here rather than imported from `policy_cfg`, because these
tests stand in for whatever writes `--policy.endpoints` — which spells them out too, having no way to
import anything. A test written against the constants would follow a rename and pin nothing.
"""

import numpy as np
import pos3
import pytest

from positronic import keys
from positronic.cfg import policy as policy_cfg
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.policy.base import SampledPolicy


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
    assert (a['replay.dataset_path'], a['replay.episode']) == (dataset, 0)
    assert (b['replay.dataset_path'], b['replay.episode']) == (dataset, 1)


def test_replay_endpoint_without_an_episode_takes_the_first(tmp_path):
    dataset = _dataset(tmp_path / 'ds', episodes=2)

    meta = _session_meta(_built(endpoints={'arm_a': {'kind': 'replay', 'dataset': dataset}}))

    assert meta['replay.episode'] == 0


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


def test_weights_still_name_endpoints_declared_as_mappings(tmp_path):
    dataset = _dataset(tmp_path / 'ds', episodes=1)

    with pytest.raises(ValueError, match=r"weights name unknown endpoints: \['arm_b'\]"):
        _built(endpoints={'arm_a': {'kind': 'replay', 'dataset': dataset}}, weights={'arm_b': 1.0})
