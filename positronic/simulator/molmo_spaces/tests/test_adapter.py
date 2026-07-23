"""Unit tests for the pi05_droid <-> MolmoSpaces adapter mapping logic.

Runs without molmo_spaces and without positronic's heavy stack: the adapter needs only the light
``positronic-client`` package, and every test here exercises the pure mapping functions, ``ChunkBuffer``,
and ``FakePolicy`` — none of which touch a framework or a server.

Run:  uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_adapter.py --no-cov
"""

from pathlib import Path

import numpy as np
import pytest
from positronic_client import keys

from positronic.simulator.molmo_spaces import adapter
from positronic.simulator.molmo_spaces.adapter import (
    NUM_ARM_JOINTS,
    ROBOTIQ_CLOSED,
    ROBOTIQ_OPEN,
    ChunkBuffer,
    FakePolicy,
    _FakeJointDelta,
    adapter_config_from_exp_config,
    molmo_obs_to_positronic,
    positronic_action_to_molmo,
)

FIXTURE = Path(__file__).parent / 'droid_obs.npz'


def _load_env_obs() -> dict:
    data = np.load(FIXTURE)
    return {
        'wrist_camera': data['wrist_camera'],
        'exo_camera_1': data['exo_camera_1'],
        'qpos': {'arm': data['qpos_arm'], 'gripper': data['qpos_gripper']},
    }


# --- import guard -----------------------------------------------------------------------------------------------


def test_module_imports_without_frameworks():
    # The pure logic must be usable even when molmo_spaces is absent (the common test/dev box).
    assert isinstance(adapter.HAS_MOLMO_SPACES, bool)
    assert callable(molmo_obs_to_positronic)
    assert callable(positronic_action_to_molmo)


# --- observation mapping ----------------------------------------------------------------------------------------


def test_obs_mapping_key_set():
    obs = molmo_obs_to_positronic(_load_env_obs(), 'pick up the cube')
    assert set(obs) == {keys.JOINTS, keys.GRIP, keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.TASK}


def test_obs_mapping_shapes_and_dtypes():
    env = _load_env_obs()
    obs = molmo_obs_to_positronic(env, 'Pick up the cube')
    assert obs[keys.JOINTS].shape == (NUM_ARM_JOINTS,) and obs[keys.JOINTS].dtype == np.float32
    assert obs[keys.GRIP].shape == (1,) and obs[keys.GRIP].dtype == np.float32
    # Frames and task text pass through untouched: model preprocessing (resize-with-pad, prompt lowercasing)
    # belongs to the server codec, wire downsizing to the inference client.
    assert np.array_equal(obs[keys.WRIST_IMAGE], env['wrist_camera'])
    assert np.array_equal(obs[keys.EXTERIOR_IMAGE], env['exo_camera_1'])
    assert obs[keys.TASK] == 'Pick up the cube'


def test_obs_mapping_accepts_batch_list_and_single_dict():
    env = _load_env_obs()
    from_dict = molmo_obs_to_positronic(env, 't')
    from_list = molmo_obs_to_positronic([env], 't')  # MolmoSpaces yields a per-env list
    assert np.array_equal(from_dict[keys.JOINTS], from_list[keys.JOINTS])
    assert np.array_equal(from_dict[keys.WRIST_IMAGE], from_list[keys.WRIST_IMAGE])


def test_obs_mapping_does_not_swap_cameras():
    obs = molmo_obs_to_positronic(_load_env_obs(), 't')
    # Fixture marks the wrist view reddish and the exterior view greenish; a swap would flip the dominant channel.
    wrist_mean = obs[keys.WRIST_IMAGE].reshape(-1, 3).mean(axis=0)
    exterior_mean = obs[keys.EXTERIOR_IMAGE].reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]  # wrist: red > green
    assert exterior_mean[1] > exterior_mean[0]  # exterior: green > red


def test_obs_mapping_resolves_benchmark_variant_camera_keys():
    # Zed-wrist / light-randomized variants replace the default camera keys; MolmoSpaces' own Pi policy
    # prefers the variant key when present, so the adapter must too (regression: hard indexing raised
    # KeyError on those benchmark observations).
    env = _load_env_obs()
    env['wrist_camera_zed_mini'] = env.pop('wrist_camera')
    env['droid_shoulder_light_randomization'] = env.pop('exo_camera_1')
    obs = molmo_obs_to_positronic(env, 't')
    wrist_mean = obs[keys.WRIST_IMAGE].reshape(-1, 3).mean(axis=0)
    exterior_mean = obs[keys.EXTERIOR_IMAGE].reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]  # the reddish wrist view still lands on the wrist key
    assert exterior_mean[1] > exterior_mean[0]

    # A variant key coexisting with the default wins, matching the upstream policy's precedence.
    both = _load_env_obs()
    both['droid_shoulder_light_randomization'] = both['wrist_camera']  # reddish, unlike exo_camera_1
    obs = molmo_obs_to_positronic(both, 't')
    exterior_mean = obs[keys.EXTERIOR_IMAGE].reshape(-1, 3).mean(axis=0)
    assert exterior_mean[0] > exterior_mean[1]

    # An explicitly configured non-default key is read as-is, never shadowed by a variant.
    custom = _load_env_obs()
    custom['my_cam'] = custom['wrist_camera']
    custom['wrist_camera_zed_mini'] = custom['exo_camera_1']  # greenish decoy
    obs = molmo_obs_to_positronic(custom, 't', wrist_key='my_cam')
    wrist_mean = obs[keys.WRIST_IMAGE].reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]


class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_adapter_config_honors_policy_config_camera_names():
    # An eval configured with non-default camera names (policy_config.camera_names, per MolmoSpaces'
    # custom-policy convention) must reach the adapter's camera keys (regression: a default AdapterConfig
    # ignored them and KeyErrored on the renamed observations).
    cfg = _Ns(policy_config=_Ns(camera_names=['randomized_zed2_analogue_1', 'wrist_camera_zed_mini']))
    adapter_cfg = adapter_config_from_exp_config(cfg)
    assert adapter_cfg.wrist_key == 'wrist_camera_zed_mini'
    assert adapter_cfg.exterior_key == 'randomized_zed2_analogue_1'

    # No camera_names declared (BasePolicyConfig doesn't define the field) -> the defaults, variant fallback intact.
    for cfg in (None, _Ns(policy_config=_Ns()), _Ns(policy_config=_Ns(camera_names=[]))):
        adapter_cfg = adapter_config_from_exp_config(cfg)
        assert adapter_cfg.wrist_key == adapter.MOLMO_WRIST_CAMERA
        assert adapter_cfg.exterior_key == adapter.MOLMO_EXTERIOR_CAMERA

    # A single-role list keeps the missing role on its default.
    adapter_cfg = adapter_config_from_exp_config(_Ns(policy_config=_Ns(camera_names=['exo_top_down'])))
    assert adapter_cfg.wrist_key == adapter.MOLMO_WRIST_CAMERA
    assert adapter_cfg.exterior_key == 'exo_top_down'

    # The policy's default path derives from the config it was constructed with. force_enable_depth is the one
    # field MolmoSpaces' BasePolicy.__init__ reads, so the stub carries it and the test also passes on boxes
    # where molmo_spaces IS installed and the real superclass path runs.
    names = ['randomized_zed2_analogue_1', 'wrist_camera_zed_mini']
    cfg = _Ns(policy_config=_Ns(camera_names=names, force_enable_depth=False))
    policy = adapter.MolmoSpacesPolicy(cfg, client=FakePolicy())
    assert policy._adapter.wrist_key == 'wrist_camera_zed_mini'
    assert policy._adapter.exterior_key == 'randomized_zed2_analogue_1'


def test_client_kwargs_honor_policy_config_remote_config():
    # MolmoSpaces' learned-policy configs carry remote_config=dict(host, port) for remotely served policies;
    # the default policy_factory(exp_config, task) path must reach that endpoint, not the localhost defaults.
    cfg = _Ns(policy_config=_Ns(remote_config={'host': 'h100.example', 'port': 9000}))
    assert adapter.client_kwargs_from_exp_config(cfg) == {'host': 'h100.example', 'port': 9000}
    # Keys make_policy_client doesn't take are filtered out.
    cfg = _Ns(policy_config=_Ns(remote_config={'host': 'h', 'checkpoint_path': '/x'}))
    assert adapter.client_kwargs_from_exp_config(cfg) == {'host': 'h'}
    # Absent or None remote_config (or no config at all) -> defaults.
    assert adapter.client_kwargs_from_exp_config(None) == {}
    assert adapter.client_kwargs_from_exp_config(_Ns(policy_config=_Ns(remote_config=None))) == {}


def test_gripper_proprio_normalization():
    closed = adapter.GRIPPER_QPOS_CLOSED

    def grip_for(qpos_val: float) -> float:
        env = _load_env_obs()
        env['qpos']['gripper'] = np.array([qpos_val, qpos_val], dtype=np.float32)
        return float(molmo_obs_to_positronic(env, 't')[keys.GRIP][0])

    assert grip_for(0.0) == 0.0
    assert abs(grip_for(closed / 2) - 0.5) < 1e-4
    assert abs(grip_for(closed) - 1.0) < 1e-6
    assert grip_for(closed * 2) == 1.0  # saturates, never exceeds 1


# --- action mapping ---------------------------------------------------------------------------------------------


def test_action_integrates_delta_onto_live_joints():
    current = np.arange(NUM_ARM_JOINTS, dtype=np.float32)
    velocities = np.full(NUM_ARM_JOINTS, 0.1, dtype=np.float32)
    action = {'robot_command': _FakeJointDelta(velocities), 'target_grip': 1.0}
    out = positronic_action_to_molmo(action, current)
    assert out['arm'].shape == (NUM_ARM_JOINTS,) and out['arm'].dtype == np.float32
    assert np.allclose(out['arm'], current + velocities)
    assert out['gripper'].shape == (1,)


def test_action_gripper_convention():
    current = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    vel = _FakeJointDelta(np.zeros(NUM_ARM_JOINTS, dtype=np.float32))

    def gripper_for(target_grip: float) -> float:
        out = positronic_action_to_molmo({'robot_command': vel, 'target_grip': target_grip}, current)
        return float(out['gripper'][0])

    assert gripper_for(1.0) == ROBOTIQ_CLOSED == 255.0
    assert gripper_for(0.0) == ROBOTIQ_OPEN == 0.0
    assert gripper_for(0.9) == ROBOTIQ_CLOSED  # binarized above 0.5
    assert gripper_for(0.1) == ROBOTIQ_OPEN


def test_action_reads_velocities_from_object_or_wire_dict():
    current = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    vel = np.linspace(-0.2, 0.2, NUM_ARM_JOINTS, dtype=np.float32)
    from_obj = positronic_action_to_molmo({'robot_command': _FakeJointDelta(vel), 'target_grip': 0.0}, current)
    from_dict = positronic_action_to_molmo({'robot_command': {'velocities': vel}, 'target_grip': 0.0}, current)
    assert np.allclose(from_obj['arm'], from_dict['arm'])
    # The base wire delivers the command as its __cmd__ envelope around the to_wire dict.
    wire_cmd = {b'__cmd__': {'type': 'joint_delta', 'velocities': vel}}
    from_wire = positronic_action_to_molmo({'robot_command': wire_cmd, 'target_grip': 0.0}, current)
    assert np.allclose(from_obj['arm'], from_wire['arm'])
    # Bytes-keyed wire form (msgpack deserialisation on some client versions keys with bytes).
    from_bytes = positronic_action_to_molmo({b'robot_command': {b'velocities': vel}, b'target_grip': 1.0}, current)
    assert np.allclose(from_obj['arm'], from_bytes['arm'])


def test_action_joint_count_mismatch_raises():
    current = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    bad = {'robot_command': _FakeJointDelta(np.zeros(6, dtype=np.float32)), 'target_grip': 0.0}
    with pytest.raises(ValueError):
        positronic_action_to_molmo(bad, current)


# --- FakePolicy -------------------------------------------------------------------------------------------------


def test_fake_policy_chunk_shape_and_range():
    policy = FakePolicy(chunk_size=8, seed=3)
    chunk = policy.new_session().infer({keys.TASK: 't'})
    assert len(chunk) == 8
    for step in chunk:
        vel = step['robot_command'].velocities
        assert vel.shape == (NUM_ARM_JOINTS,)
        assert np.all(np.abs(vel) <= adapter.MAX_JOINT_DELTA + 1e-6)
        assert step['target_grip'] in (0.0, 1.0)


def test_fake_policy_is_deterministic_per_seed():
    a = FakePolicy(seed=7).new_session().infer({keys.TASK: 't'})
    b = FakePolicy(seed=7).new_session().infer({keys.TASK: 't'})
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa['robot_command'].velocities, sb['robot_command'].velocities)
        assert sa['target_grip'] == sb['target_grip']


def test_fake_policy_zero_mode_holds():
    chunk = FakePolicy(mode='zero').new_session().infer({keys.TASK: 't'})
    for step in chunk:
        assert np.array_equal(step['robot_command'].velocities, np.zeros(NUM_ARM_JOINTS))
        assert step['target_grip'] == 0.0


# --- ChunkBuffer ------------------------------------------------------------------------------------------------


class _CountingSession:
    def __init__(self, chunk_size: int):
        self.calls = 0
        self._chunk_size = chunk_size

    def infer(self, obs):
        self.calls += 1
        return [
            {
                'robot_command': _FakeJointDelta(np.full(NUM_ARM_JOINTS, self.calls, dtype=np.float32)),
                'target_grip': 0.0,
            }
            for _ in range(self._chunk_size)
        ]


def test_chunk_buffer_replays_one_per_tick_then_requeries():
    session = _CountingSession(chunk_size=3)
    buf = ChunkBuffer(session)
    first = [buf.next({}) for _ in range(3)]
    assert session.calls == 1  # one chunk covered three ticks
    fourth = buf.next({})
    assert session.calls == 2  # drained -> re-queried
    assert float(first[0]['robot_command'].velocities[0]) == 1.0
    assert float(fourth['robot_command'].velocities[0]) == 2.0


def test_chunk_buffer_empty_chunk_raises():
    class _Empty:
        def infer(self, obs):
            return []

    buf = ChunkBuffer(_Empty())
    with pytest.raises(RuntimeError):
        buf.next({})


def test_chunk_buffer_drops_trailing_horizon_marker():
    # A real droid chunk is 8 actions + a window-end entry carrying only `timestamp`; consumed as an action the
    # marker KeyErrors on `robot_command`, so the buffer must drop it and re-query rather than replay it.
    vel = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    chunk = [{'robot_command': _FakeJointDelta(vel), 'target_grip': 0.0, 'timestamp': i / 15} for i in range(8)]
    chunk.append({'timestamp': 8 / 15})
    calls = {'n': 0}

    class _Session:
        def infer(self, obs):
            calls['n'] += 1
            return list(chunk)

    buf = ChunkBuffer(_Session())
    for _ in range(8):
        assert 'robot_command' in buf.next({})
    assert calls['n'] == 1
    buf.next({})  # 9th tick: marker dropped, buffer re-queries instead of replaying it
    assert calls['n'] == 2


# --- end to end (server-free) -----------------------------------------------------------------------------------


def test_end_to_end_fake_pipeline():
    env = _load_env_obs()
    client = FakePolicy(mode='random', seed=1, chunk_size=8)
    buffer = ChunkBuffer(client.new_session())

    pos_obs = molmo_obs_to_positronic(env, 'pick up the cube')
    current_arm = np.asarray(env['qpos']['arm'], dtype=np.float32)
    action = buffer.next(pos_obs)
    molmo_action = positronic_action_to_molmo(action, current_arm)

    assert set(molmo_action) == {'arm', 'gripper'}
    assert molmo_action['arm'].shape == (NUM_ARM_JOINTS,)
    assert molmo_action['gripper'].shape == (1,)
    assert molmo_action['gripper'][0] in (ROBOTIQ_OPEN, ROBOTIQ_CLOSED)
    expected_arm = current_arm + action['robot_command'].velocities
    assert np.allclose(molmo_action['arm'], expected_arm)
