"""Tests for DreamZeroObservationCodec."""

import numpy as np
import pytest

from positronic.vendors.dreamzero.codecs import DreamZeroObservationCodec


class TestDreamZeroObservationCodec:
    @pytest.fixture
    def sample_inputs(self):
        return {
            'robot_state.q': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            'grip': np.array([0.5]),
            'image.wrist': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'image.exterior': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'image.exterior2': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'task': 'pick up the cube',
        }

    def test_encode_basic(self, sample_inputs):
        codec = DreamZeroObservationCodec()
        result = codec.encode(sample_inputs)

        assert 'observation/joint_position' in result
        assert 'observation/gripper_position' in result
        assert 'observation/wrist_image_left' in result
        assert 'observation/exterior_image_0_left' in result
        assert 'observation/exterior_image_1_left' in result
        assert 'prompt' in result

        assert result['observation/joint_position'].shape == (7,)
        assert result['observation/gripper_position'].shape == (1,)
        assert np.allclose(result['observation/joint_position'], sample_inputs['robot_state.q'])
        assert result['prompt'] == 'pick up the cube'

    def test_encode_image_resize(self, sample_inputs):
        codec = DreamZeroObservationCodec()
        result = codec.encode(sample_inputs)

        # Images should be resized to 320x180 (W×H) → array shape (180, 320, 3)
        assert result['observation/wrist_image_left'].shape == (180, 320, 3)
        assert result['observation/exterior_image_0_left'].shape == (180, 320, 3)
        assert result['observation/exterior_image_1_left'].shape == (180, 320, 3)

    def test_encode_missing_task(self, sample_inputs):
        del sample_inputs['task']
        codec = DreamZeroObservationCodec()
        result = codec.encode(sample_inputs)

        assert 'prompt' not in result

    def test_encode_missing_third_camera(self, sample_inputs):
        del sample_inputs['image.exterior2']
        codec = DreamZeroObservationCodec()
        result = codec.encode(sample_inputs)

        # Missing camera should be padded with zeros
        assert result['observation/exterior_image_1_left'].shape == (180, 320, 3)
        assert np.all(result['observation/exterior_image_1_left'] == 0)

    def test_dummy_encoded_shape(self):
        codec = DreamZeroObservationCodec()
        result = codec.dummy_encoded()

        assert result['observation/joint_position'].shape == (7,)
        assert result['observation/gripper_position'].shape == (1,)
        assert result['observation/wrist_image_left'].shape == (180, 320, 3)
        assert result['observation/exterior_image_0_left'].shape == (180, 320, 3)
        assert result['observation/exterior_image_1_left'].shape == (180, 320, 3)
        assert result['prompt'] == 'warmup'

    def test_meta(self):
        assert DreamZeroObservationCodec().meta == {'image_sizes': (320, 180)}

    def test_custom_camera_keys(self, sample_inputs):
        sample_inputs['cam1'] = sample_inputs.pop('image.wrist')
        sample_inputs['cam2'] = sample_inputs.pop('image.exterior')
        sample_inputs['cam3'] = sample_inputs.pop('image.exterior2')

        codec = DreamZeroObservationCodec(wrist_camera='cam1', exterior_camera_1='cam2', exterior_camera_2='cam3')
        result = codec.encode(sample_inputs)

        assert result['observation/wrist_image_left'].shape == (180, 320, 3)
        assert result['observation/exterior_image_0_left'].shape == (180, 320, 3)
        assert result['observation/exterior_image_1_left'].shape == (180, 320, 3)

    def test_non_square_input_image(self, sample_inputs):
        sample_inputs['image.wrist'] = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        codec = DreamZeroObservationCodec()
        result = codec.encode(sample_inputs)

        assert result['observation/wrist_image_left'].shape == (180, 320, 3)
