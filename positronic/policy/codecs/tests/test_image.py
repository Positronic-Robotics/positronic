import json

import cv2
import numpy as np
import pytest
from PIL import Image as PilImage
from positronic_model_server import serialization
from positronic_model_server.spec import LATEST, NAME, component

from pimm.time import RECEIVED_WALL, RECEIVED_WORLD
from positronic.cfg.codecs import image_observation
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.policy import keys as policy_keys
from positronic.policy import spec, training
from positronic.policy.codecs import LEROBOT_FEATURES, Codec, EncodeImages, Resample, ResizeMode, TransformImages


@pytest.mark.parametrize(
    'shape, resized, pad',
    [((6, 8), (6, 4), (0, 1)), ((2, 4), (6, 3), (0, 1)), ((6, 6), (6, 6), (0, 0)), ((8, 3), (2, 6), (2, 0))],
)
def test_image_observations_preserve_pixels_training_timelines_and_metadata(shape, resized, pad):
    frame = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    definition = image_observation(images={'video': ('camera', (6, 6))})
    description = json.loads(json.dumps(definition.to_spec()))
    codec = spec.from_spec(description)
    assert isinstance(codec, Codec)
    assert codec.to_spec() == description
    expected = np.zeros((6, 6, 3), dtype=np.uint8)
    x, y = pad
    w, h = resized
    expected[y : y + h, x : x + w] = np.asarray(PilImage.fromarray(frame).resize(resized, PilImage.Resampling.BILINEAR))

    encoded = codec.encode({'camera': frame, 'unused': 1})
    assert set(encoded) == {'video'}
    np.testing.assert_array_equal(encoded['video'], expected, strict=True)

    episode = EpisodeContainer(
        {
            'camera': DummySignal([[10, 100], [20, 200]], [frame, frame], timelines=(RECEIVED_WORLD, RECEIVED_WALL)),
            'unused': 1,
        },
        meta={'label': 'test'},
    )
    recipe = training.image_observation(images={'video': ('camera', (6, 6))})
    columns = recipe(episode)
    assert set(columns) == {'video'}
    assert columns.meta == episode.meta
    assert columns['video'].timelines == (RECEIVED_WORLD, RECEIVED_WALL)
    assert list(columns['video'].timestamps(RECEIVED_WORLD)) == [10, 20]
    assert list(columns['video'].timestamps(RECEIVED_WALL)) == [100, 200]
    np.testing.assert_array_equal(columns['video'].values(), np.stack([expected, expected]), strict=True)
    assert codec.meta == {}
    assert recipe.meta == {
        LEROBOT_FEATURES: {'video': {'shape': (6, 6, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'}}
    }
    assert codec.decode({'action': 1}) == {}


@pytest.mark.parametrize('as_list', [False, True])
def test_image_observations_accept_lists_and_noncontiguous_frames(as_list):
    frame = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, ::-1]
    data = frame.tolist() if as_list else frame
    codec = image_observation(images={'image': ('camera', (6, 4))})
    expected = frame.astype(np.int64) if as_list else frame
    np.testing.assert_array_equal(codec.encode({'camera': data})['image'], expected, strict=True)


def test_image_observations_require_rgb_images():
    codec = image_observation(images={'image': ('camera', (8, 6))})
    with pytest.raises(KeyError, match='camera'):
        codec.encode({})
    with pytest.raises(ValueError, match='HWC with 3 channels'):
        codec.encode({'camera': np.zeros((8, 8), dtype=np.uint8)})


def test_image_observations_support_different_camera_sizes_and_no_images():
    codec = image_observation(images={'front': ('a', (8, 6)), 'wrist': ('b', (4, 4))})
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    encoded = codec.encode({'a': frame, 'b': frame})
    assert encoded['front'].shape == (6, 8, 3)
    assert encoded['wrist'].shape == (4, 4, 3)
    assert codec.meta == {}
    empty = image_observation(images={})
    assert empty.encode({'unused': 1}) == {}
    assert empty.meta == {}
    assert len(empty.training_encoder(EpisodeContainer({}))) == 0
    assert empty.training_encoder.meta == {LEROBOT_FEATURES: {}}


@pytest.mark.parametrize('mode', list(ResizeMode))
@pytest.mark.parametrize('resample', list(Resample))
@pytest.mark.parametrize('stacked', [False, True])
@pytest.mark.parametrize('shape', [(8, 12), (2, 3)])
def test_transforms_use_the_selected_interpolation_for_frames_and_history(mode, resample, stacked, shape):
    frames = np.random.default_rng(30).integers(0, 256, (25, *shape, 3), dtype=np.uint8)
    expected = []
    if mode is ResizeMode.STRETCH:
        size = (6, 6)
    elif mode is ResizeMode.BOUND and shape == (2, 3):
        size = (3, 2)
    else:
        size = (6, 4)
    for frame in frames:
        pixels = (
            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if resample is Resample.AREA
            else np.asarray(PilImage.fromarray(frame).resize(size, PilImage.Resampling.BILINEAR))
        )
        expected.append(np.pad(pixels, ((1, 1), (0, 0), (0, 0))) if mode is ResizeMode.PAD else pixels)
    codec = TransformImages(6, 6, mode=mode, resample=resample)
    restored = spec.from_spec(json.loads(json.dumps(codec.to_spec())))
    assert isinstance(restored, TransformImages)
    assert restored.to_spec() == codec.to_spec()
    inputs = {'video': {'wrist': frames if stacked else frames[0]}, 'state': np.array([1, 2]), 'task': 'pick'}
    output = restored.encode(inputs)
    np.testing.assert_array_equal(output['video']['wrist'], np.stack(expected) if stacked else expected[0], strict=True)
    assert output['state'] is inputs['state']
    assert output['task'] == 'pick'
    assert inputs['video']['wrist'].shape == ((25, *shape, 3) if stacked else (*shape, 3))


@pytest.mark.parametrize('mode', list(ResizeMode))
def test_exact_size_frame_is_unchanged(mode):
    frame = np.random.default_rng(5).integers(0, 256, (4, 6, 3), dtype=np.uint8)
    assert TransformImages(6, 4, mode=mode).encode({'cam': frame})['cam'] is frame


@pytest.mark.parametrize('resample', list(Resample))
def test_image_training_and_inference_share_the_selected_filter(resample):
    frame = np.random.default_rng(2).integers(0, 256, (5, 9, 3), dtype=np.uint8)
    images = {'front': ('camera', (12, 12))}
    definition = image_observation(images=images, resample=resample)
    codec = spec.from_spec(json.loads(json.dumps(definition.to_spec())))
    assert isinstance(codec, Codec)
    recipe = training.image_observation(images, resample)
    episode = EpisodeContainer({'camera': DummySignal([10, 20], [frame, frame])})
    expected = codec.encode({'camera': frame})['front']
    np.testing.assert_array_equal(recipe(episode)['front'].values(), np.stack([expected, expected]), strict=True)
    np.testing.assert_array_equal(definition.training_encoder(episode)['front'].values()[0], expected, strict=True)


def test_explicit_image_paths_preserve_names_other_images_and_containers():
    selected = np.zeros((8, 12, 3), dtype=np.uint8)
    untouched = np.ones((8, 12, 3), dtype=np.uint8)
    inputs = {'cameras': (selected, untouched), 'literal.key': selected}
    codec = TransformImages(6, 6, mode=ResizeMode.PAD, paths=[['cameras', 0], ['literal.key']])
    restored = spec.from_spec(json.loads(json.dumps(codec.to_spec())))
    assert isinstance(restored, TransformImages)
    output = restored.encode(inputs)
    assert set(output) == set(inputs)
    assert isinstance(output['cameras'], tuple)
    assert output['cameras'][0].shape == (6, 6, 3)
    assert output['cameras'][1] is untouched
    assert output['literal.key'].shape == (6, 6, 3)
    assert inputs['cameras'][0] is selected
    assert TransformImages(6, 6, paths=[], mode=ResizeMode.BOUND).encode(inputs) is inputs
    with pytest.raises(KeyError):
        codec.encode({})
    with pytest.raises(ValueError, match='HWC'):
        TransformImages(6, 6, paths=[['state']], mode=ResizeMode.BOUND).encode({'state': [1, 2]})


def _image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestTransformImages:
    def test_bounds_every_image(self):
        result = TransformImages(64, 48, mode=ResizeMode.BOUND).encode({
            'cam_a': _image(480, 640),
            'cam_b': _image(240, 320),
            'state': np.array([1.0]),
        })
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], np.array([1.0]))

    def test_aspect_is_kept_and_images_only_shrink(self):
        result = TransformImages(160, 160, mode=ResizeMode.BOUND).encode({
            'wide': _image(480, 640),
            'small': _image(24, 32),
        })
        assert result['wide'].shape == (120, 160, 3)
        assert result['small'].shape == (24, 32, 3)

    def test_image_within_bound_is_the_same_object(self):
        img = _image(48, 64)
        assert TransformImages(64, 48, mode=ResizeMode.BOUND).encode({'cam': img})['cam'] is img

    def test_stacked_frames_are_bounded_per_frame(self):
        stack = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        assert TransformImages(64, 48, mode=ResizeMode.BOUND).encode({'cam': stack})['cam'].shape == (3, 48, 64, 3)

    def test_a_threaded_stack_scales_to_the_same_pixels_as_one_thread(self):
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, size=(TransformImages._PARALLEL_FROM + 4, 480, 640, 3), dtype=np.uint8)
        codec = TransformImages(64, 48, mode=ResizeMode.BOUND)
        one_at_a_time = np.stack([codec.encode({'cam': frame})['cam'] for frame in stack])
        np.testing.assert_array_equal(codec.encode({'cam': stack})['cam'], one_at_a_time)

    def test_a_single_usable_cpu_stays_serial(self, monkeypatch):
        """A pool wins nothing on a single core, and costs threads to raise."""
        monkeypatch.setattr(TransformImages, '_usable_cpus', staticmethod(lambda: 1))
        codec = TransformImages(64, 48, mode=ResizeMode.BOUND)
        assert codec._workers(codec._PARALLEL_FROM + 4) == 1

    def test_the_pool_is_bounded_by_the_cpus_the_process_may_run_on(self, monkeypatch):
        monkeypatch.setattr(TransformImages, '_usable_cpus', staticmethod(lambda: 2))
        codec = TransformImages(64, 48, mode=ResizeMode.BOUND)
        assert codec._workers(codec._MAX_WORKERS * 4) == 2

    def test_a_stack_under_the_parallel_bar_still_scales(self):
        stack = np.zeros((TransformImages._PARALLEL_FROM - 1, 480, 640, 3), dtype=np.uint8)
        assert TransformImages(64, 48, mode=ResizeMode.BOUND).encode({'cam': stack})['cam'].shape[1:] == (48, 64, 3)

    def test_nested_images_are_reached(self):
        result = TransformImages(64, 48, mode=ResizeMode.BOUND).encode({
            'video': {'cam': _image(480, 640)},
            'seq': [_image(480, 640)],
        })
        assert result['video']['cam'].shape == (48, 64, 3)
        assert result['seq'][0].shape == (48, 64, 3)

    def test_non_image_values_pass_through(self):
        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = TransformImages(64, 48, mode=ResizeMode.BOUND).encode(obs)
        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True

    def test_actions_pass_through_untouched(self):
        actions = [{'target_grip': 0.5}, {'target_grip': 1.0}]
        assert TransformImages(64, 48, mode=ResizeMode.BOUND).decode(actions) == actions

    def test_an_rtc_prefix_passes_through_while_the_images_shrink(self):
        prefix = [{'target_grip': 0.5}, {'target_grip': 1.0}]
        received = []

        def model(obs):
            received.append(obs)
            return []

        codec = TransformImages(64, 48, mode=ResizeMode.BOUND)
        codec.wrap(model)({'cam': _image(480, 640), policy_keys.ACTION_PREFIX: prefix})

        assert received[0][policy_keys.ACTION_PREFIX] == prefix
        assert received[0]['cam'].shape == (48, 64, 3)

    def test_training_encoder_refuses(self):
        with pytest.raises(NotImplementedError, match='full-resolution'):
            _ = TransformImages(64, 48, mode=ResizeMode.BOUND).training_encoder

    def test_survives_a_wire_round_trip(self):
        rebuilt = spec.from_spec(TransformImages(64, 48, mode=ResizeMode.BOUND).to_spec())
        assert isinstance(rebuilt, TransformImages)
        assert rebuilt.encode({'cam': _image(480, 640)})['cam'].shape == (48, 64, 3)

    def test_a_spec_without_a_filter_uses_bilinear(self):
        description = component(TransformImages.WIRE_NAME, version=1, width=64, height=48, mode='bound')
        restored = spec.from_spec(description)
        assert isinstance(restored, TransformImages)
        assert restored.to_spec() == TransformImages(64, 48, mode=ResizeMode.BOUND).to_spec()

    @pytest.mark.parametrize('bounds', [{}, {'width': 64}, {'height': 48}])
    def test_transform_specs_require_both_dimensions(self, bounds):
        with pytest.raises(TypeError, match='required positional argument'):
            spec.from_spec(component(TransformImages.WIRE_NAME, mode='bound', **bounds))

    def test_transform_specs_require_a_known_mode(self):
        with pytest.raises(TypeError, match='mode'):
            spec.from_spec(component(TransformImages.WIRE_NAME, width=64, height=48))
        with pytest.raises(ValueError, match='unknown'):
            spec.from_spec(component(TransformImages.WIRE_NAME, width=64, height=48, mode='unknown'))

    def test_a_legacy_spec_without_bounds_takes_the_standard_bound(self):
        restored = spec.from_spec({NAME: 'restrict_image_size'})
        assert isinstance(restored, TransformImages)
        assert restored.encode({'cam': _image(1080, 1920)})['cam'].shape == (360, 640, 3)

    def test_area_survives_a_wire_round_trip(self):
        description = TransformImages(64, 48, resample=Resample.AREA, mode=ResizeMode.BOUND).to_spec()
        assert description == component(
            TransformImages.WIRE_NAME,
            version=1,
            width=64,
            height=48,
            resample=Resample.AREA.value,
            mode=ResizeMode.BOUND.value,
            paths=None,
        )
        rebuilt = spec.from_spec(description)
        assert isinstance(rebuilt, TransformImages)
        assert rebuilt.to_spec() == description

    @staticmethod
    def _textured(h, w):
        y, x = np.mgrid[0:h, 0:w]
        image = np.stack([127 + 100 * np.sin(x / 7.0) * np.cos(y / 11.0), 255 * x / w, 255 * y / h], axis=-1)
        return np.clip(image, 0, 255).astype(np.uint8)

    @pytest.mark.parametrize('version', [LATEST, 1])
    @pytest.mark.parametrize('bounds', [{}, {'width': 64}, {'height': 48}, {'width': 64, 'height': 48}])
    @pytest.mark.parametrize('resample', [None, *Resample])
    def test_server_specs_preserve_defaults_and_explicit_filters(self, version, bounds, resample):
        args = bounds if resample is None else {**bounds, 'resample': resample.value}
        rebuilt = spec.from_spec(component('restrict_image_size', version=version, **args))
        assert isinstance(rebuilt, TransformImages)
        expected = TransformImages(
            width=bounds.get('width', 640),
            height=bounds.get('height', 640),
            resample=Resample.BILINEAR if resample is None else resample,
            mode=ResizeMode.BOUND,
        )
        assert rebuilt.to_spec() == expected.to_spec()
        frame = self._textured(720, 960)
        np.testing.assert_array_equal(rebuilt.encode({'cam': frame})['cam'], expected.encode({'cam': frame})['cam'])

    @pytest.mark.parametrize('version', [LATEST, 1])
    def test_an_unknown_resample_in_a_spec_is_refused(self, version):
        description = component('restrict_image_size', version=version, resample='bicubic')
        with pytest.raises(ValueError, match='bicubic'):
            spec.from_spec(description)

    @pytest.mark.parametrize('bound', [(320, 320), (160, 160), (64, 48)])
    def test_area_gives_the_bilinear_shape_and_near_pixels(self, bound):
        image = self._textured(480, 640)
        bilinear = TransformImages(*bound, mode=ResizeMode.BOUND).encode({'cam': image})['cam']
        area = TransformImages(*bound, resample=Resample.AREA, mode=ResizeMode.BOUND).encode({'cam': image})['cam']
        assert area.shape == bilinear.shape and area.dtype == bilinear.dtype
        assert np.abs(area.astype(int) - bilinear.astype(int)).mean() < 2

    def test_area_scales_a_frame_that_is_not_contiguous(self):
        image = self._textured(480, 640)[:, ::-1]
        codec = TransformImages(64, 48, resample=Resample.AREA, mode=ResizeMode.BOUND)
        expected = codec.encode({'cam': np.ascontiguousarray(image)})['cam']
        np.testing.assert_array_equal(codec.encode({'cam': image})['cam'], expected)

    def test_a_threaded_area_stack_scales_to_the_same_pixels_as_one_thread(self):
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, size=(TransformImages._PARALLEL_FROM + 4, 480, 640, 3), dtype=np.uint8)
        codec = TransformImages(64, 48, resample=Resample.AREA, mode=ResizeMode.BOUND)
        one_at_a_time = np.stack([codec.encode({'cam': frame})['cam'] for frame in stack])
        np.testing.assert_array_equal(codec.encode({'cam': stack})['cam'], one_at_a_time)


class TestEncodeImages:
    def test_an_rtc_prefix_passes_through_unchanged(self):
        prefix = [{'action': np.array([0.1, 0.2], dtype=np.float32)}]
        received = []

        def model(obs):
            received.append(obs[policy_keys.ACTION_PREFIX])
            return []

        EncodeImages().wrap(model)({policy_keys.ACTION_PREFIX: prefix})

        np.testing.assert_array_equal(received[0][0]['action'], prefix[0]['action'])

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
