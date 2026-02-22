"""Tests for GrootObservationCodec."""

import numpy as np
import pytest

from positronic import geom
from positronic.vendors.gr00t.codecs import GrootObservationCodec

RotRep = geom.Rotation.Representation


class TestGrootObservationCodec:
    """Tests for GrootObservationCodec training and inference modes."""

    @pytest.fixture
    def sample_inputs(self):
        """Sample raw inputs for inference encoding."""
        return {
            'robot_state.ee_pose': np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),  # xyz + quat (w,x,y,z)
            'grip': np.array([0.5]),
            'robot_state.q': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            'image.wrist': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'image.exterior': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'task': 'pick up the cube',
        }

    # --- Inference encoding tests ---

    def test_encode_basic(self, sample_inputs):
        """Test basic inference encoding without rotation conversion."""
        codec = GrootObservationCodec(rotation_rep=None, include_joints=False)
        result = codec.encode(sample_inputs)

        assert 'video' in result
        assert 'state' in result
        assert 'language' in result

        assert 'wrist_image' in result['video']
        assert 'exterior_image_1' in result['video']
        assert result['video']['wrist_image'].shape == (1, 1, 224, 224, 3)
        assert result['video']['exterior_image_1'].shape == (1, 1, 224, 224, 3)

        assert 'ee_pose' in result['state']
        assert 'grip' in result['state']
        assert 'joint_position' not in result['state']
        assert result['state']['ee_pose'].shape == (1, 1, 7)
        assert result['state']['grip'].shape == (1, 1, 1)

        assert result['language']['annotation.language.language_instruction'] == [['pick up the cube']]

    def test_encode_with_rot6d(self, sample_inputs):
        """Test inference encoding with rot6d conversion."""
        codec = GrootObservationCodec(rotation_rep=RotRep.ROT6D, include_joints=False)
        result = codec.encode(sample_inputs)

        assert result['state']['ee_pose'].shape == (1, 1, 9)

        ee_pose = result['state']['ee_pose'][0, 0]
        assert np.allclose(ee_pose[:3], sample_inputs['robot_state.ee_pose'][:3])

        expected_rot6d = geom.Rotation.from_quat(sample_inputs['robot_state.ee_pose'][3:7]).as_rot6d
        assert np.allclose(ee_pose[3:], expected_rot6d, atol=1e-6)

    def test_encode_with_joints(self, sample_inputs):
        """Test inference encoding with joint positions."""
        codec = GrootObservationCodec(rotation_rep=None, include_joints=True)
        result = codec.encode(sample_inputs)

        assert 'joint_position' in result['state']
        assert result['state']['joint_position'].shape == (1, 1, 7)
        assert np.allclose(result['state']['joint_position'][0, 0], sample_inputs['robot_state.q'])

    def test_encode_with_rot6d_and_joints(self, sample_inputs):
        """Test inference encoding with both rot6d and joints."""
        codec = GrootObservationCodec(rotation_rep=RotRep.ROT6D, include_joints=True)
        result = codec.encode(sample_inputs)

        assert result['state']['ee_pose'].shape == (1, 1, 9)
        assert result['state']['grip'].shape == (1, 1, 1)
        assert result['state']['joint_position'].shape == (1, 1, 7)

    def test_encode_missing_task(self, sample_inputs):
        """Test inference encoding handles missing task gracefully."""
        del sample_inputs['task']
        codec = GrootObservationCodec()
        result = codec.encode(sample_inputs)

        assert result['language']['annotation.language.language_instruction'] == [['']]

    # --- Rot6d conversion correctness tests ---

    def test_rot6d_identity_quaternion(self, sample_inputs):
        """Test rot6d conversion with identity quaternion."""
        sample_inputs['robot_state.ee_pose'] = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        codec = GrootObservationCodec(rotation_rep=RotRep.ROT6D)
        result = codec.encode(sample_inputs)

        ee_pose = result['state']['ee_pose'][0, 0]
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert np.allclose(ee_pose[3:], expected_rot6d, atol=1e-6)

    def test_rot6d_90deg_rotation(self, sample_inputs):
        """Test rot6d conversion with 90 degree rotation around Z."""
        quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        sample_inputs['robot_state.ee_pose'] = np.array([1.0, 2.0, 3.0, *quat])

        codec = GrootObservationCodec(rotation_rep=RotRep.ROT6D)
        result = codec.encode(sample_inputs)

        ee_pose = result['state']['ee_pose'][0, 0]
        expected_rot6d = geom.Rotation.from_quat(quat).as_rot6d
        assert np.allclose(ee_pose[3:], expected_rot6d, atol=1e-6)

    # --- Output key tests ---

    def test_output_keys_basic(self):
        """Test that codec outputs correct keys for training."""
        codec = GrootObservationCodec(rotation_rep=None, include_joints=False)

        expected = {'ee_pose', 'grip', 'wrist_image', 'exterior_image_1', 'task'}
        assert set(codec._derive_transforms.keys()) == expected

    def test_output_keys_with_joints(self):
        """Test that codec includes joint_position when enabled."""
        codec = GrootObservationCodec(rotation_rep=RotRep.ROT6D, include_joints=True)

        assert 'joint_position' in codec._derive_transforms

    # --- Metadata tests ---

    def test_training_meta(self):
        """Test that training metadata is computed from constructor params."""
        codec = GrootObservationCodec(rotation_rep=RotRep.ROT6D, include_joints=True)
        meta = codec._training_meta

        assert 'gr00t_modality' in meta
        assert 'lerobot_features' in meta
        assert 'joint_position' in meta['lerobot_features']
        assert meta['lerobot_features']['ee_pose']['shape'] == (9,)

    def test_training_meta_no_joints(self):
        """Test that training metadata excludes joints when not enabled."""
        codec = GrootObservationCodec(rotation_rep=None, include_joints=False)
        meta = codec._training_meta

        assert 'joint_position' not in meta['lerobot_features']
        assert meta['lerobot_features']['ee_pose']['shape'] == (7,)

    # --- Edge cases ---

    def test_non_square_input_image(self, sample_inputs):
        """Test that non-square images are properly resized with padding."""
        sample_inputs['image.wrist'] = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        codec = GrootObservationCodec()
        result = codec.encode(sample_inputs)

        assert result['video']['wrist_image'].shape == (1, 1, 224, 224, 3)

    def test_custom_image_size(self, sample_inputs):
        """Test custom image size."""
        codec = GrootObservationCodec(image_size=(128, 128))
        result = codec.encode(sample_inputs)

        assert result['video']['wrist_image'].shape == (1, 1, 128, 128, 3)
        assert result['video']['exterior_image_1'].shape == (1, 1, 128, 128, 3)

    @pytest.mark.parametrize(
        'rotation_rep, include_joints', [(None, False), (RotRep.ROT6D, False), (None, True), (RotRep.ROT6D, True)]
    )
    def test_dummy_encoded_shape(self, rotation_rep, include_joints):
        """Test that dummy_encoded() produces valid encoded observations for all variants."""
        codec = GrootObservationCodec(rotation_rep=rotation_rep, include_joints=include_joints)
        result = codec.dummy_encoded()
        assert 'video' in result
        assert 'state' in result
        assert 'language' in result

    def test_custom_camera_keys(self, sample_inputs):
        """Test custom camera key mapping."""
        sample_inputs['cam1'] = sample_inputs.pop('image.wrist')
        sample_inputs['cam2'] = sample_inputs.pop('image.exterior')

        codec = GrootObservationCodec(wrist_camera='cam1', exterior_camera='cam2')
        result = codec.encode(sample_inputs)

        assert result['video']['wrist_image'].shape == (1, 1, 224, 224, 3)
        assert result['video']['exterior_image_1'].shape == (1, 1, 224, 224, 3)
