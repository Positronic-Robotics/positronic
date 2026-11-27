from unittest.mock import MagicMock

import configuronic as cfn
import numpy as np
import pytest
from positronic_model_server import server_wire
from positronic_wire import wire

from positronic import geom, keys
from positronic.cfg import codecs
from positronic.dataset.transforms import TransformedDataset
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.models import FRANKA_HAND_FRAME, GRIPPER_SITE, PANDA_HAND_FRAME, parse_ee_frames
from positronic.offboard.protocol import deserialise
from positronic.offboard.spec import PolicyDeployment
from positronic.policy import keys as policy_keys
from positronic.policy.codecs.observation import TASK_FIELD
from positronic.policy.processors import ChunkedSchedule, TrajectoryPlayer
from positronic.policy.sequential import Sequential

torch = pytest.importorskip('torch')
pytest.importorskip('lerobot')

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402

from positronic.offboard.server import PolicyServer  # noqa: E402
from positronic.vendors.lerobot_0_3_3 import codecs as lerobot_codecs  # noqa: E402
from positronic.vendors.lerobot_0_3_3 import server as lerobot_server  # noqa: E402
from positronic.vendors.lerobot_0_3_3.policy import LerobotModel, warm_observation  # noqa: E402
from positronic.vendors.lerobot_0_3_3.tests import records  # noqa: E402

STATE_FEATURE = 'observation.state'
CAMERA_FEATURE = 'observation.images.left'


def _act_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            STATE_FEATURE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            CAMERA_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 320)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )


def _loads_a_mock_policy(monkeypatch) -> None:
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path, prefix='': ['42'])
    monkeypatch.setattr(lerobot_server.pos3, 'download', lambda path: path)
    policy = MagicMock(spec=lerobot_server.ACTPolicy, config=_act_config())
    monkeypatch.setattr(lerobot_server.ACTPolicy, 'from_pretrained', lambda *_args, **_kwargs: policy)
    # A mock cannot answer an inference, but the warm observation is still built from the loaded policy,
    # so the load reaches no checkpoint on disk.
    monkeypatch.setattr(lerobot_server, 'warmup', lambda *_args, **_kwargs: None)


def test_handshake_metadata_names_the_loaded_checkpoint(monkeypatch):
    """``checkpoint_path`` is in the metadata, since sampling keys on it."""
    _loads_a_mock_policy(monkeypatch)
    model = lerobot_server.act_model(checkpoints_dir='s3://bucket/exp', device='cpu')
    assert model.meta() == {
        policy_keys.CHECKPOINT_ID: '42',
        'type': 'act',
        'checkpoint_path': 's3://bucket/exp/checkpoints/42/pretrained_model',
        'experiment_name': 'exp',
        'device': 'cpu',
    }
    model.close()


def test_the_metadata_states_the_license_of_the_checkpoint(monkeypatch):
    _loads_a_mock_policy(monkeypatch)
    model = lerobot_server.act_model(checkpoints_dir='s3://bucket/exp', device='cpu', license='internal')
    assert model.meta()[policy_keys.LICENSE] == 'internal'
    model.close()


def test_act_sessions_reuse_full_chunk_prediction_without_resetting_the_model():
    policy = MagicMock()
    policy.to.return_value = policy
    policy.predict_action_chunk.return_value = torch.zeros((1, 2, 7))
    model = LerobotModel(policy, device='cpu')
    for session_id in ('first', 'second'):
        chunk = model({TASK_FIELD: 'stack'}, session_id=session_id)
        assert len(chunk) == 2
        assert all(step['action'].shape == (7,) for step in chunk)
        model.end_session(session_id)
    assert policy.predict_action_chunk.call_count == 2
    policy.reset.assert_not_called()
    policy.select_action.assert_not_called()
    model.close()


def _make_server(monkeypatch, checkpoint: str | None) -> tuple[PolicyServer, MagicMock]:
    """A server whose model build makes no backbone, and the mock that stands in for the model class."""
    build = MagicMock(side_effect=lambda _backbone, _device, extra_meta: MagicMock(**{'meta.return_value': extra_meta}))
    monkeypatch.setattr(lerobot_server, 'LerobotModel', build)
    monkeypatch.setattr(lerobot_server.pos3, 'download', lambda path: path)
    monkeypatch.setattr(lerobot_server, 'warmup', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lerobot_server.ACTPolicy, 'from_pretrained', lambda *_args, **_kwargs: MagicMock())
    model_cfg = lerobot_server.act_model.override(checkpoints_dir='s3://bucket/exp', checkpoint=checkpoint)
    return PolicyServer(
        model_cfg, PolicyDeployment(local=Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=15)))
    ), build


@pytest.mark.asyncio
async def test_lerobot_server_uses_configured_checkpoint(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41', '42'])
    server, _build = _make_server(monkeypatch, checkpoint='42')
    server._load()
    connection = MagicMock(
        spec=server_wire.ServerConnection,
        peer='test',
        query_params=[],
        served_address=server_wire.ServedHostPort('localhost', 8000),
    )
    connection.receive.side_effect = wire.PeerDisconnected('Session ended')
    await server._serve_session(connection)

    ready = deserialise(connection.send.await_args_list[0].args[0])
    assert ready['status'] == 'ready'
    assert ready['meta'][policy_keys.CHECKPOINT_ID] == '42'


def test_lerobot_server_rejects_missing_configured_checkpoint_at_startup(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41'])
    server, build = _make_server(monkeypatch, checkpoint='42')

    with pytest.raises(ValueError) as excinfo:
        server._load()

    assert 'Configured checkpoint not found: 42' in str(excinfo.value)
    assert "Available: ['41']" in str(excinfo.value)
    build.assert_not_called()


def test_warmup_observation_matches_the_features_the_policy_declares():
    obs = warm_observation(_act_config())

    assert obs[STATE_FEATURE].shape == (8,)
    # The model callable accepts channels-last images.
    assert obs[CAMERA_FEATURE].shape == (224, 320, 3)
    assert obs[TASK_FIELD] == ''


def _pipeline(name: str) -> cfn.Config:
    return lerobot_server.COMMANDS[name].kwargs['pipeline']


@pytest.mark.parametrize('name', ['serve', 'ee', 'joints', 'ee_traj', 'ee_flip', 'ee_bimanual'])
def test_a_pose_pipeline_that_states_no_frame_fails_to_build(name):
    with pytest.raises(TypeError, match='ee_frame'):
        _pipeline(name).instantiate()


@pytest.mark.parametrize('name', ['joints_traj', 'joints_ik', 'joints_ik_sim'])
def test_a_joint_pipeline_builds_with_no_frame(name):
    assert roboarm_keys.EE_FRAMES not in _pipeline(name).instantiate().meta()


@pytest.mark.parametrize(
    'name, offset', [('phail', FRANKA_HAND_FRAME), ('sim_stack', PANDA_HAND_FRAME), ('demo', PANDA_HAND_FRAME)]
)
def test_a_bound_checkpoint_states_the_frame_it_was_trained_in(name, offset):
    frame = parse_ee_frames(_pipeline(name).instantiate().meta()[roboarm_keys.EE_FRAMES])[keys.EE_POSE]
    assert (frame.site, frame.offset) == (GRIPPER_SITE, offset)


QUAT = geom.Rotation.Representation.QUAT


def _two_arm_observation(sample: int, grips: dict[str, float] | None = None) -> dict:
    grips = grips or {arm: records.arm_grip(arm, sample) for arm in keys.BIMANUAL_ARMS}
    return {
        **{keys.arm_key(keys.EE_POSE, arm): records.arm_pose(arm, sample) for arm in keys.BIMANUAL_ARMS},
        **{keys.arm_channel(keys.GRIP, arm): grip for arm, grip in grips.items()},
        **dict.fromkeys(records.TWO_ARM_CAMERAS, records.IMAGE),
        keys.TASK: records.TASK,
    }


def _two_arm_action(left_grip: float, right_grip: float) -> dict:
    pose = [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
    return {'action': np.array([*pose, left_grip, *pose, right_grip], dtype=np.float32)}


def test_the_two_arm_codec_and_its_server_pipeline_give_one_state_and_one_command_for_each_arm(tmp_path):
    # A frame with an offset, so that a pose that the frame change does not move gives a different state or command.
    frame = codecs.yam_grasp_site_frame
    training = lerobot_codecs.ee_bimanual.override(ee_frame=frame).instantiate().training_encoder
    record = records.write_record(tmp_path, keys.BIMANUAL_ARMS, records.TWO_ARM_CAMERAS)
    episode = TransformedDataset(record, training)[0]
    codec = _pipeline('ee_bimanual').override(ee_frame=frame).instantiate().codec
    one_arm = _pipeline('ee').override(ee_frame=frame).instantiate().codec

    states = [state for state, _ts in episode[STATE_FEATURE]]
    assert len(states) == records.SAMPLES
    for sample, state in enumerate(states):
        np.testing.assert_allclose(state, codec.encode(_two_arm_observation(sample))[STATE_FEATURE], atol=1e-6)
        # The state of each arm is the state that the one-arm pipeline gives for the pose and the grip of that arm.
        for i, arm in enumerate(keys.BIMANUAL_ARMS):
            alone = one_arm.encode({
                keys.EE_POSE: records.arm_pose(arm, sample),
                keys.GRIP: records.arm_grip(arm, sample),
                **dict.fromkeys(records.ONE_ARM_CAMERAS, records.IMAGE),
                keys.TASK: records.TASK,
            })[STATE_FEATURE]
            np.testing.assert_allclose(state[i * 8 : (i + 1) * 8], alone, atol=1e-6)
    for sample, (action, _ts) in enumerate(episode['action']):
        decoded = codec.decode({'action': action})
        for arm in keys.BIMANUAL_ARMS:
            pose = decoded[keys.arm_channel(keys.ROBOT_COMMAND, arm)].pose.as_vector(QUAT)
            np.testing.assert_allclose(pose, records.arm_pose(arm, sample + 1), atol=1e-6)
            assert decoded[keys.arm_channel(keys.TARGET_GRIP, arm)] == pytest.approx(records.arm_grip(arm, sample + 1))


def test_the_two_arm_pipeline_states_the_frame_of_the_poses_of_each_arm():
    meta = _pipeline('ee_bimanual').override(ee_frame=codecs.gripper_frame).instantiate().meta()
    frames = parse_ee_frames(meta[roboarm_keys.EE_FRAMES])
    for arm in keys.BIMANUAL_ARMS:
        assert frames[keys.arm_key(keys.EE_POSE, arm)].site == GRIPPER_SITE
        assert frames[keys.arm_key(keys.TARGET_EE_POSE, arm)].site == GRIPPER_SITE


def test_the_two_arm_pipeline_flips_the_grip_of_each_arm():
    codec = _pipeline('ee_bimanual').override(ee_frame=codecs.gripper_frame, flip_grip=True).instantiate().codec
    observation = _two_arm_observation(0, {keys.LEFT_ARM: 0.9, keys.RIGHT_ARM: 0.2})

    state = codec.encode(observation)[STATE_FEATURE]
    decoded = codec.decode(_two_arm_action(0.9, 0.2))

    np.testing.assert_allclose(state[[7, 15]], [0.1, 0.8], atol=1e-6)
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == pytest.approx(0.1)
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == pytest.approx(0.8)


def test_the_two_arm_pipeline_binarizes_the_target_grip_of_each_arm():
    grips = tuple(keys.arm_channel(keys.GRIP, arm) for arm in keys.BIMANUAL_ARMS)
    pipeline = _pipeline('ee_bimanual').override(ee_frame=codecs.gripper_frame, binarize_grip=grips)

    decoded = pipeline.instantiate().codec.decode(_two_arm_action(0.9, 0.2))

    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == 1.0
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == 0.0
