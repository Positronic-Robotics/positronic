"""Unit tests for ``MolmoAdapter``: the raw env-server payload -> canonical embodiment contract.

Runs without molmo_spaces (the env server lives in its own interpreter); it needs positronic, which is where
the adapter runs. Exercises the observation mapping against a synthetic raw payload (``droid_obs.npz``), the
terminal, and the reset token.

Run:  uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_adapter.py --no-cov
"""

from pathlib import Path

import numpy as np

from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import MolmoAdapter

FIXTURE = Path(__file__).parent / 'droid_obs.npz'
CAMERA_DICT = {'image.wrist': mapping.MOLMO_WRIST_CAMERA, 'image.exterior': mapping.MOLMO_EXTERIOR_CAMERA}


def _payload() -> dict:
    return dict(np.load(FIXTURE).items())


def test_observations_assemble_robot_state():
    payload = _payload()
    obs = MolmoAdapter(CAMERA_DICT).observations(payload)
    state = obs['robot_state']
    assert np.allclose(state.q, payload['joint_pos'])
    assert np.allclose(state.dq, payload['joint_vel'])
    assert np.allclose(state.ee_pose.translation, payload['eef_pos'])
    assert np.allclose(state.ee_pose.rotation.as_quat, payload['eef_quat'])  # wxyz round-trips
    assert obs['grip'] == 0.5


def test_observations_camera_passthrough_no_swap():
    payload = _payload()
    obs = MolmoAdapter(CAMERA_DICT).observations(payload)
    # Frames pass through untouched (no resize/flip — the codec/client own preprocessing/transport).
    assert np.array_equal(obs['image.wrist'].array, payload['wrist_camera'])
    assert np.array_equal(obs['image.exterior'].array, payload['exo_camera_1'])
    # Fixture marks wrist reddish, exterior greenish; a swap would flip the dominant channel.
    wrist_mean = obs['image.wrist'].array.reshape(-1, 3).mean(axis=0)
    exterior_mean = obs['image.exterior'].array.reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]
    assert exterior_mean[1] > exterior_mean[0]


def test_observations_resolve_benchmark_variant_camera():
    # A Zed-wrist benchmark replaces the default key; the adapter must still land the reddish wrist view on
    # image.wrist (regression: hard indexing KeyErrored on those observations).
    payload = _payload()
    payload['wrist_camera_zed_mini'] = payload.pop('wrist_camera')
    obs = MolmoAdapter(CAMERA_DICT).observations(payload)
    wrist_mean = obs['image.wrist'].array.reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]


def test_terminal_reports_success_only_when_done():
    adapter = MolmoAdapter(CAMERA_DICT)
    assert adapter.terminal({'done': True, 'success': True}) == {'eval.success': True}
    assert adapter.terminal({'done': True, 'success': False}) == {'eval.success': False}
    assert adapter.terminal({'done': False, 'success': False}) is None


def test_reset_token_carries_episode_and_seed():
    adapter = MolmoAdapter(CAMERA_DICT)
    assert adapter.reset_token({'eval.episode_index': 3, 'eval.seed': 7}) == {'episode_index': 3, 'seed': 7}
    # An absent seed falls back to the spec's own (None here).
    assert adapter.reset_token({'eval.episode_index': 2}) == {'episode_index': 2, 'seed': None}
