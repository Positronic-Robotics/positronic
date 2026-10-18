import numpy as np
import pytest
from positronic_model_server import serialization
from positronic_model_server.spec import LATEST, NAME, component

from positronic.policy import spec
from positronic.policy.codecs import EncodeImages, Resample, RestrictImageSize


def _image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestRestrictImageSize:
    def test_bounds_every_image(self):
        result = RestrictImageSize(64, 48).encode({
            'cam_a': _image(480, 640),
            'cam_b': _image(240, 320),
            'state': np.array([1.0]),
        })
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], np.array([1.0]))

    def test_defaults_to_the_standard_bound(self):
        assert RestrictImageSize().encode({'cam': _image(1080, 1920)})['cam'].shape == (360, 640, 3)

    def test_aspect_is_kept_and_images_only_shrink(self):
        result = RestrictImageSize(160, 160).encode({'wide': _image(480, 640), 'small': _image(24, 32)})
        assert result['wide'].shape == (120, 160, 3)
        assert result['small'].shape == (24, 32, 3)

    def test_image_within_bound_is_the_same_object(self):
        img = _image(48, 64)
        assert RestrictImageSize(64, 48).encode({'cam': img})['cam'] is img

    def test_stacked_frames_are_bounded_per_frame(self):
        stack = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        assert RestrictImageSize(64, 48).encode({'cam': stack})['cam'].shape == (3, 48, 64, 3)

    def test_a_threaded_stack_scales_to_the_same_pixels_as_one_thread(self):
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, size=(RestrictImageSize._PARALLEL_FROM + 4, 480, 640, 3), dtype=np.uint8)
        codec = RestrictImageSize(64, 48)
        one_at_a_time = np.stack([codec.encode({'cam': frame})['cam'] for frame in stack])
        np.testing.assert_array_equal(codec.encode({'cam': stack})['cam'], one_at_a_time)

    def test_a_single_usable_cpu_stays_serial(self, monkeypatch):
        """A pool wins nothing on a single core, and costs threads to raise."""
        monkeypatch.setattr(RestrictImageSize, '_usable_cpus', staticmethod(lambda: 1))
        codec = RestrictImageSize(64, 48)
        assert codec._workers(codec._PARALLEL_FROM + 4) == 1

    def test_the_pool_is_bounded_by_the_cpus_the_process_may_run_on(self, monkeypatch):
        monkeypatch.setattr(RestrictImageSize, '_usable_cpus', staticmethod(lambda: 2))
        codec = RestrictImageSize(64, 48)
        assert codec._workers(codec._MAX_WORKERS * 4) == 2

    def test_a_stack_under_the_parallel_bar_still_scales(self):
        stack = np.zeros((RestrictImageSize._PARALLEL_FROM - 1, 480, 640, 3), dtype=np.uint8)
        assert RestrictImageSize(64, 48).encode({'cam': stack})['cam'].shape[1:] == (48, 64, 3)

    def test_nested_images_are_reached(self):
        result = RestrictImageSize(64, 48).encode({'video': {'cam': _image(480, 640)}, 'seq': [_image(480, 640)]})
        assert result['video']['cam'].shape == (48, 64, 3)
        assert result['seq'][0].shape == (48, 64, 3)

    def test_non_image_values_pass_through(self):
        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = RestrictImageSize(64, 48).encode(obs)
        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True

    def test_actions_pass_through_untouched(self):
        actions = [{'target_grip': 0.5}, {'target_grip': 1.0}]
        assert RestrictImageSize(64, 48).decode(actions) == actions

    def test_training_encoder_refuses(self):
        with pytest.raises(NotImplementedError, match='full-resolution'):
            _ = RestrictImageSize(64, 48).training_encoder

    def test_survives_a_wire_round_trip(self):
        rebuilt = spec.from_spec(RestrictImageSize(64, 48).to_spec())
        assert isinstance(rebuilt, RestrictImageSize)
        assert rebuilt.encode({'cam': _image(480, 640)})['cam'].shape == (48, 64, 3)

    def test_a_spec_without_a_filter_uses_bilinear(self):
        description = component(RestrictImageSize.WIRE_NAME, version=1, width=64, height=48)
        restored = spec.from_spec(description)
        assert isinstance(restored, RestrictImageSize)
        assert restored.to_spec() == RestrictImageSize(64, 48).to_spec()

    def test_a_v1_spec_without_bounds_takes_the_standard_bound(self):
        restored = spec.from_spec({NAME: RestrictImageSize.WIRE_NAME})
        assert isinstance(restored, RestrictImageSize)
        assert restored.encode({'cam': _image(1080, 1920)})['cam'].shape == (360, 640, 3)

    def test_area_survives_a_wire_round_trip(self):
        description = RestrictImageSize(64, 48, resample=Resample.AREA).to_spec()
        assert description == component(
            RestrictImageSize.WIRE_NAME, version=1, width=64, height=48, resample=Resample.AREA.value
        )
        rebuilt = spec.from_spec(description)
        assert isinstance(rebuilt, RestrictImageSize)
        assert rebuilt.to_spec() == description

    @pytest.mark.parametrize('version', [LATEST, 1])
    @pytest.mark.parametrize('bounds', [{}, {'width': 64}, {'height': 48}, {'width': 64, 'height': 48}])
    @pytest.mark.parametrize('resample', [None, *Resample])
    def test_server_specs_preserve_defaults_and_explicit_filters(self, version, bounds, resample):
        args = bounds if resample is None else {**bounds, 'resample': resample.value}
        rebuilt = spec.from_spec(component(RestrictImageSize.WIRE_NAME, version=version, **args))
        assert isinstance(rebuilt, RestrictImageSize)
        expected = RestrictImageSize(**bounds, resample=Resample.BILINEAR if resample is None else resample)
        assert rebuilt.to_spec() == expected.to_spec()
        frame = self._textured(720, 960)
        np.testing.assert_array_equal(rebuilt.encode({'cam': frame})['cam'], expected.encode({'cam': frame})['cam'])

    @pytest.mark.parametrize('version', [LATEST, 1])
    def test_an_unknown_resample_in_a_spec_is_refused(self, version):
        description = component(RestrictImageSize.WIRE_NAME, version=version, resample='bicubic')
        with pytest.raises(ValueError, match='bicubic'):
            spec.from_spec(description)

    @staticmethod
    def _textured(h, w):
        y, x = np.mgrid[0:h, 0:w]
        image = np.stack([127 + 100 * np.sin(x / 7.0) * np.cos(y / 11.0), 255 * x / w, 255 * y / h], axis=-1)
        return np.clip(image, 0, 255).astype(np.uint8)

    @pytest.mark.parametrize('bound', [(320, 320), (160, 160), (64, 48)])
    def test_area_gives_the_bilinear_shape_and_near_pixels(self, bound):
        image = self._textured(480, 640)
        bilinear = RestrictImageSize(*bound).encode({'cam': image})['cam']
        area = RestrictImageSize(*bound, resample=Resample.AREA).encode({'cam': image})['cam']
        assert area.shape == bilinear.shape and area.dtype == bilinear.dtype
        assert np.abs(area.astype(int) - bilinear.astype(int)).mean() < 2

    def test_area_scales_a_frame_that_is_not_contiguous(self):
        image = self._textured(480, 640)[:, ::-1]
        codec = RestrictImageSize(64, 48, resample=Resample.AREA)
        expected = codec.encode({'cam': np.ascontiguousarray(image)})['cam']
        np.testing.assert_array_equal(codec.encode({'cam': image})['cam'], expected)

    def test_a_threaded_area_stack_scales_to_the_same_pixels_as_one_thread(self):
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, size=(RestrictImageSize._PARALLEL_FROM + 4, 480, 640, 3), dtype=np.uint8)
        codec = RestrictImageSize(64, 48, resample=Resample.AREA)
        one_at_a_time = np.stack([codec.encode({'cam': frame})['cam'] for frame in stack])
        np.testing.assert_array_equal(codec.encode({'cam': stack})['cam'], one_at_a_time)


class TestEncodeImages:
    @pytest.mark.parametrize('shape', [(8, 12, 3), (2, 8, 12, 3), (2, 3, 8, 12, 3), (0, 8, 12, 3)])
    def test_automatic_encoding_reaches_nested_images_and_preserves_dimensions(self, shape):
        image = np.full(shape, 140, dtype=np.uint8)
        obs = {'video': {'cameras': [image]}, 'frames': (image,), 'task': 'pick cube'}
        encoded = EncodeImages().encode(obs)

        assert isinstance(encoded['video']['cameras'][0], dict)
        assert isinstance(encoded['frames'], tuple)
        assert isinstance(encoded['frames'][0], dict)
        restored = serialization.deserialise(serialization.serialise(encoded))
        for decoded in (restored['video']['cameras'][0], restored['frames'][0]):
            assert decoded.shape == image.shape
            np.testing.assert_allclose(decoded, image, atol=2)
        assert restored['task'] == obs['task']
        assert obs['video']['cameras'][0] is image
        assert obs['frames'][0] is image

    @pytest.mark.parametrize(
        'value',
        [
            np.ones((8, 12, 3), dtype=np.float32),
            np.ones((8, 12, 3), dtype=np.uint16),
            np.ones((3, 8, 12), dtype=np.uint8),
            np.ones((8, 12, 4), dtype=np.uint8),
            np.ones((8, 12), dtype=np.uint8),
            np.ones(3, dtype=np.uint8),
            np.ones((0, 12, 3), dtype=np.uint8),
            b'image bytes',
        ],
        ids=['float', 'uint16', 'channels-first', 'rgba', 'grayscale', 'vector', 'empty-height', 'bytes'],
    )
    def test_automatic_encoding_preserves_values_outside_the_image_rule(self, value):
        assert EncodeImages().encode({'state': value})['state'] is value

    @pytest.mark.parametrize('paths, compressed', [(None, {'camera'}), ([['selected', 0]], {'selected'}), ([], set())])
    def test_selection_and_quality_survive_the_component_spec(self, paths, compressed):
        codec = EncodeImages(paths, quality=73)
        rebuilt = spec.from_spec(codec.to_spec())
        assert isinstance(rebuilt, EncodeImages)
        assert rebuilt.to_spec() == codec.to_spec()
        image = np.full((8, 12, 3), 140, dtype=np.uint8)
        selected = image.astype(np.float32)
        obs = {'camera': image, 'selected': [selected]}
        encoded = rebuilt.encode(obs)
        assert isinstance(encoded['camera'], dict) == ('camera' in compressed)
        assert isinstance(encoded['selected'][0], dict) == ('selected' in compressed)
        restored = serialization.deserialise(serialization.serialise(encoded))
        np.testing.assert_allclose(restored['camera'], image, atol=2)
        np.testing.assert_allclose(restored['selected'][0], selected, atol=2)
        assert obs['selected'][0] is selected
        assert rebuilt.decode(obs) is obs

    @pytest.mark.parametrize('quality', [-1, 101, True, 1.5])
    def test_invalid_quality_is_rejected_before_encoding(self, quality):
        with pytest.raises(ValueError, match='JPEG quality'):
            EncodeImages(quality=quality)
