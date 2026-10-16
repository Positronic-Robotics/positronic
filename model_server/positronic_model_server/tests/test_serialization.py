"""Native values and explicit image encodings in either direction."""

from types import MappingProxyType

import msgpack
import numpy as np
import pytest
from positronic_model_server import serialization


@pytest.mark.parametrize('path', [(), ('nested', 0)])
def test_selected_image_paths_preserve_the_source_and_other_arrays(path):
    image = np.full((1, 2, 8, 10, 3), 125, dtype=np.uint8)
    state = np.full(image.shape, 1.25, dtype=np.float32)
    source = {'nested': [image], 'state': state} if path else image
    encoded = serialization.encode_images(source, [serialization.JpegEncoding(path)])
    restored = serialization.deserialise(serialization.serialise(encoded))
    selected = restored['nested'][0] if path else restored
    assert selected.shape == image.shape
    np.testing.assert_allclose(selected, image, atol=2)
    if path:
        assert isinstance(source, dict)
        assert source['nested'][0] is image
        np.testing.assert_array_equal(restored['state'], state)


def test_a_missing_image_path_is_not_silently_ignored():
    with pytest.raises(KeyError, match='missing'):
        serialization.encode_images({}, [serialization.JpegEncoding(('missing',))])


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'uint8', 'bool', '>f4'])
@pytest.mark.parametrize('shape', [(), (0, 3), (2, 4, 3)])
def test_arrays_preserve_dtype_shape_and_values(dtype, shape):
    array = np.ones(shape, dtype=dtype)
    restored = serialization.deserialise(serialization.serialise(array))
    assert restored.dtype == array.dtype
    assert restored.shape == shape
    np.testing.assert_array_equal(restored, array)


def test_native_result_has_no_robot_command_interpretation():
    actions = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    info = {'robot_command': {'type': 'model-owned'}, b'__cmd__': {'native': True}}
    result = serialization.deserialise(serialization.serialise((MappingProxyType({'actions': actions}), info)))
    np.testing.assert_array_equal(result[0]['actions'], actions)
    assert result[1] == info


def test_noncontiguous_arrays_and_numpy_scalars():
    source = np.arange(36, dtype=np.int16).reshape(6, 6)[::2, ::2]
    decoded = serialization.deserialise(serialization.serialise([source, np.float32(0.5), np.int64(7)]))
    np.testing.assert_array_equal(decoded[0], source)
    assert type(decoded[1]) is np.float32
    assert type(decoded[2]) is np.int64


@pytest.mark.parametrize('dtype', ['object', 'complex64', [('x', 'int32')]])
def test_unsupported_numpy_types_fail(dtype):
    with pytest.raises(ValueError, match='Unsupported dtype'):
        serialization.serialise(np.zeros(2, dtype=dtype))


@pytest.mark.parametrize('shape', [(8, 12, 3), (3, 8, 12, 3), (2, 3, 8, 12, 3), (2, 1, 3, 8, 12, 3)])
def test_nested_images_restore_all_dimensions_and_frame_order(shape):
    image = np.empty(shape, dtype=np.uint8)
    for index, frame in enumerate(image.reshape(-1, *shape[-3:])):
        frame[:] = index * 30
    payload = {'nested': [serialization.encode_jpeg(image)], 'state': image}
    restored = serialization.deserialise(serialization.serialise(payload))
    assert restored['nested'][0].shape == shape
    assert restored['nested'][0].dtype == np.uint8
    np.testing.assert_allclose(restored['nested'][0], image, atol=4)
    np.testing.assert_array_equal(restored['state'], image)
    # A receiver can also return the image through the same serializer.
    response = serialization.deserialise(serialization.serialise(serialization.encode_jpeg(restored['nested'][0])))
    assert response.shape == shape
    np.testing.assert_allclose(response, image, atol=8)


@pytest.mark.parametrize('shape', [(0, 8, 12, 3), (2, 0, 8, 12, 3)])
def test_empty_image_batches_keep_their_shape(shape):
    encoded = serialization.encode_jpeg(np.empty(shape, dtype=np.uint8))
    restored = serialization.deserialise(serialization.serialise(encoded))
    assert restored.shape == shape
    assert restored.dtype == np.uint8


# rules-allow: hardcoded-keys — these literals pin the bytes existing v1/v2 decoders understand.
@pytest.mark.parametrize('shape', [(8, 12, 3), (2, 8, 12, 3)])
def test_legacy_image_marker_is_unchanged(shape):
    encoded = serialization.encode_jpeg(np.full(shape, 70, dtype=np.uint8))
    wire = msgpack.unpackb(serialization.serialise(encoded))
    assert wire.keys() == {b'__jpeg__', b'frames', b'ndim'}
    assert wire[b'__jpeg__'] is True
    assert wire[b'ndim'] == len(shape)
    assert len(wire[b'frames']) == (1 if len(shape) == 3 else shape[0])


# rules-allow: hardcoded-keys — these literals pin the legacy array marker independently of its writer.
def test_legacy_array_marker_decodes():
    wire = msgpack.packb({b'__ndarray__': True, b'data': b'\x01\x00\x02\x00', b'dtype': '<i2', b'shape': [2]})
    np.testing.assert_array_equal(serialization.deserialise(wire), np.array([1, 2], dtype='<i2'))


@pytest.mark.parametrize('shape', [(8, 12), (8, 12, 4), (0, 12, 3)])
def test_invalid_image_shape_is_rejected(shape):
    with pytest.raises(ValueError, match='JPEG images'):
        serialization.encode_jpeg(np.empty(shape, dtype=np.uint8))


@pytest.mark.parametrize('quality', [-1, 101, True, 4.5])
def test_invalid_quality_is_rejected(quality):
    with pytest.raises(ValueError, match='JPEG quality'):
        serialization.encode_jpeg(np.zeros((8, 12, 3), dtype=np.uint8), quality)


def _frames(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (count, 16, 24, 3), dtype=np.uint8)


def _jpegs(image: np.ndarray, encoded=()) -> list[bytes]:
    marker = serialization.encode_jpeg(image, encoded=encoded)
    return [frame.jpeg for frame in serialization.encoded_frames(image, marker)]


def test_equal_leading_frames_reuse_their_jpeg_bytes():
    earlier = _frames(3)
    encoded = serialization.encoded_frames(earlier, serialization.encode_jpeg(earlier))
    reused = _jpegs(np.concatenate([earlier[:2], _frames(1, seed=1)]), encoded)
    assert reused[0] is encoded[0].jpeg and reused[1] is encoded[1].jpeg and reused[2] != encoded[2].jpeg
    # A frame after the first one that differs is encoded again, even when it equals an earlier frame.
    shifted = _jpegs(np.concatenate([_frames(1, seed=2), earlier[1:]]), encoded)
    assert shifted[1] is not encoded[1].jpeg and shifted[1] == encoded[1].jpeg


def test_encoded_frames_pair_each_frame_with_its_jpeg_bytes():
    stack = _frames(2)
    encoded = serialization.encoded_frames(stack, serialization.encode_jpeg(stack))
    assert [frame.pixels.shape for frame in encoded] == [(16, 24, 3)] * 2
    np.testing.assert_array_equal(encoded[1].pixels, stack[1])


def test_a_held_frame_travels_as_its_digest_and_decodes_from_the_held_frames():
    stack = _frames(3)
    held: dict[bytes, np.ndarray] = {}
    pushed = serialization.serialise({'pushed': serialization.encode_jpeg(stack[:2])})
    full = msgpack.unpackb(pushed, object_hook=lambda obj: serialization.unpack(obj, {}, held))['pushed']
    marker = serialization.encode_jpeg(stack)
    digests = serialization.frame_digests(serialization.encode_jpeg(stack[:2]))
    referenced, count = serialization.reference_frames({'image': marker}, digests)
    assert count == 2
    assert len(serialization.serialise(referenced)) < len(serialization.serialise({'image': marker}))
    restored = msgpack.unpackb(
        serialization.serialise(referenced), object_hook=lambda obj: serialization.unpack(obj, held, {})
    )['image']
    np.testing.assert_array_equal(restored[:2], full)
    np.testing.assert_array_equal(
        restored, serialization.deserialise(serialization.serialise({'image': marker}))['image']
    )


def test_the_last_frame_of_a_stack_and_a_single_image_travel_in_full():
    stack = _frames(2)
    every_frame = {serialization.frame_digest(data) for data in _jpegs(stack)}
    _, stack_count = serialization.reference_frames(serialization.encode_jpeg(stack), every_frame)
    single = serialization.encode_jpeg(stack[0])
    _, single_count = serialization.reference_frames(single, serialization.frame_digests(single))
    assert (stack_count, single_count) == (1, 0)


def test_a_digest_the_decoder_does_not_hold_is_an_unknown_frame():
    stack = _frames(2)
    marker = serialization.encode_jpeg(stack)
    referenced, _ = serialization.reference_frames(marker, serialization.frame_digests(marker))
    with pytest.raises(serialization.UnknownFrame):
        serialization.deserialise(serialization.serialise(referenced))
