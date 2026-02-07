"""Tests for GrootObservationEncoder."""

import numpy as np
import pytest

from positronic import geom
from positronic.vendors.gr00t.codecs import GrootObservationEncoder

ROT_REP = geom.Rotation.Representation


class TestGrootObservationEncoder:
    """Tests for GrootObservationEncoder training and inference modes."""

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
        encoder = GrootObservationEncoder(rotation_rep=None, include_joints=False)
        result = encoder.encode(sample_inputs)

        # Check structure
        assert 'video' in result
        assert 'state' in result
        assert 'language' in result

        # Check video keys and shapes
        assert 'wrist_image' in result['video']
        assert 'exterior_image_1' in result['video']
        assert result['video']['wrist_image'].shape == (1, 1, 224, 224, 3)
        assert result['video']['exterior_image_1'].shape == (1, 1, 224, 224, 3)

        # Check state keys and shapes (7D ee_pose without rot6d)
        assert 'ee_pose' in result['state']
        assert 'grip' in result['state']
        assert 'joint_position' not in result['state']
        assert result['state']['ee_pose'].shape == (1, 1, 7)
        assert result['state']['grip'].shape == (1, 1, 1)

        # Check language
        assert result['language']['annotation.language.language_instruction'] == [['pick up the cube']]

    def test_encode_with_rot6d(self, sample_inputs):
        """Test inference encoding with rot6d conversion."""
        encoder = GrootObservationEncoder(rotation_rep=ROT_REP.ROT6D, include_joints=False)
        result = encoder.encode(sample_inputs)

        # Check state shape (9D ee_pose with rot6d)
        assert result['state']['ee_pose'].shape == (1, 1, 9)

        # Verify rot6d conversion correctness
        ee_pose = result['state']['ee_pose'][0, 0]
        assert np.allclose(ee_pose[:3], sample_inputs['robot_state.ee_pose'][:3])  # xyz unchanged

        # Verify rot6d is valid (should be first two rows of identity matrix for identity quat)
        expected_rot6d = geom.Rotation.from_quat(sample_inputs['robot_state.ee_pose'][3:7]).as_rot6d
        assert np.allclose(ee_pose[3:], expected_rot6d, atol=1e-6)

    def test_encode_with_joints(self, sample_inputs):
        """Test inference encoding with joint positions."""
        encoder = GrootObservationEncoder(rotation_rep=None, include_joints=True)
        result = encoder.encode(sample_inputs)

        assert 'joint_position' in result['state']
        assert result['state']['joint_position'].shape == (1, 1, 7)
        assert np.allclose(result['state']['joint_position'][0, 0], sample_inputs['robot_state.q'])

    def test_encode_with_rot6d_and_joints(self, sample_inputs):
        """Test inference encoding with both rot6d and joints."""
        encoder = GrootObservationEncoder(rotation_rep=ROT_REP.ROT6D, include_joints=True)
        result = encoder.encode(sample_inputs)

        assert result['state']['ee_pose'].shape == (1, 1, 9)
        assert result['state']['grip'].shape == (1, 1, 1)
        assert result['state']['joint_position'].shape == (1, 1, 7)

    def test_encode_missing_task(self, sample_inputs):
        """Test inference encoding handles missing task gracefully."""
        del sample_inputs['task']
        encoder = GrootObservationEncoder()
        result = encoder.encode(sample_inputs)

        assert result['language']['annotation.language.language_instruction'] == [['']]

    # --- Rot6d conversion correctness tests ---

    def test_rot6d_identity_quaternion(self, sample_inputs):
        """Test rot6d conversion with identity quaternion."""
        # Identity quaternion in wxyz format: w=1, x=0, y=0, z=0
        sample_inputs['robot_state.ee_pose'] = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        encoder = GrootObservationEncoder(rotation_rep=ROT_REP.ROT6D)
        result = encoder.encode(sample_inputs)

        ee_pose = result['state']['ee_pose'][0, 0]
        # For identity rotation, rot6d should be [1,0,0, 0,1,0] (first two rows of identity matrix)
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert np.allclose(ee_pose[3:], expected_rot6d, atol=1e-6)

    def test_rot6d_90deg_rotation(self, sample_inputs):
        """Test rot6d conversion with 90 degree rotation around Z."""
        # 90 deg rotation around Z: quat = (cos(45°), 0, 0, sin(45°)) = (0.707, 0, 0, 0.707)
        quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        sample_inputs['robot_state.ee_pose'] = np.array([1.0, 2.0, 3.0, *quat])

        encoder = GrootObservationEncoder(rotation_rep=ROT_REP.ROT6D)
        result = encoder.encode(sample_inputs)

        ee_pose = result['state']['ee_pose'][0, 0]

        # Verify using geom library
        expected_rot6d = geom.Rotation.from_quat(quat).as_rot6d
        assert np.allclose(ee_pose[3:], expected_rot6d, atol=1e-6)

    # --- Output key tests ---

    def test_output_keys_basic(self):
        """Test that encoder outputs correct keys for training."""
        encoder = GrootObservationEncoder(rotation_rep=None, include_joints=False)

        # Check derive transform keys (used in training)
        assert 'ee_pose' in encoder._transforms
        assert 'grip' in encoder._transforms
        assert 'wrist_image' in encoder._transforms
        assert 'exterior_image_1' in encoder._transforms
        assert 'joint_position' not in encoder._transforms

    def test_output_keys_with_joints(self):
        """Test that encoder includes joint_position when enabled."""
        encoder = GrootObservationEncoder(rotation_rep=ROT_REP.ROT6D, include_joints=True)

        assert 'ee_pose' in encoder._transforms
        assert 'grip' in encoder._transforms
        assert 'joint_position' in encoder._transforms

    # --- Metadata tests ---

    def test_metadata_property(self):
        """Test metadata getter/setter."""
        encoder = GrootObservationEncoder()
        assert encoder.meta == {}

        encoder.meta = {'test': 'value'}
        assert encoder.meta == {'test': 'value'}

    # --- Edge cases ---

    def test_non_square_input_image(self, sample_inputs):
        """Test that non-square images are properly resized with padding."""
        sample_inputs['image.wrist'] = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        encoder = GrootObservationEncoder()
        result = encoder.encode(sample_inputs)

        assert result['video']['wrist_image'].shape == (1, 1, 224, 224, 3)

    def test_custom_image_size(self, sample_inputs):
        """Test custom image size."""
        encoder = GrootObservationEncoder(image_size=(128, 128))
        result = encoder.encode(sample_inputs)

        assert result['video']['wrist_image'].shape == (1, 1, 128, 128, 3)
        assert result['video']['exterior_image_1'].shape == (1, 1, 128, 128, 3)

    @pytest.mark.parametrize(
        'rotation_rep, include_joints', [(None, False), (ROT_REP.ROT6D, False), (None, True), (ROT_REP.ROT6D, True)]
    )
    def test_dummy_input_roundtrip(self, rotation_rep, include_joints):
        """Test that encode(dummy_input()) succeeds for all encoder variants."""
        encoder = GrootObservationEncoder(rotation_rep=rotation_rep, include_joints=include_joints)
        result = encoder.encode(encoder.dummy_input())
        assert 'video' in result
        assert 'state' in result

    def test_custom_camera_keys(self, sample_inputs):
        """Test custom camera key mapping."""
        sample_inputs['cam1'] = sample_inputs.pop('image.wrist')
        sample_inputs['cam2'] = sample_inputs.pop('image.exterior')

        encoder = GrootObservationEncoder(wrist_camera='cam1', exterior_camera='cam2')
        result = encoder.encode(sample_inputs)

        assert result['video']['wrist_image'].shape == (1, 1, 224, 224, 3)
        assert result['video']['exterior_image_1'].shape == (1, 1, 224, 224, 3)
