"""Time the frame path of one Dream-1 call at a given upload size, on this machine.

The rig bounds each camera's window to the upload size and JPEG-encodes it, and this server decodes it, cuts it to
`codecs.FRAME_SIZE`, blurs faces and encodes it again for DreamAPI. The probe runs each step on a window of
`codecs.WINDOW_FRAMES` frames of `source` for each of the three cameras, and prints the median time of each step and the
bytes the rig sends. The network time is not measured: scale the bytes by the rate of a real run.

Usage
  uv run --locked --extra dream python -m positronic.vendors.dream.upload_probe --source=frame.jpg
  uv run --locked --extra dream python -m positronic.vendors.dream.upload_probe --source=frame.jpg \\
      --upload_width=640 --upload_height=400
"""

import statistics
import time
from collections.abc import Callable
from typing import Any

import configuronic as cfn
import cv2
import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, deserialise, encode_jpeg, serialise
from shared.faceblur import FaceBlurrer

from positronic import keys
from positronic.policy.codecs import ResizeMode, TransformImages
from positronic.vendors.dream import codecs, server


def _median_ms(step: Callable[[], Any], runs: int) -> tuple[float, Any]:
    durations, result = [], None
    for _ in range(runs):
        start = time.perf_counter()
        result = step()
        durations.append((time.perf_counter() - start) * 1e3)
    return statistics.median(durations), result


@cfn.config(upload_width=server.UPLOAD_SIZE[0], upload_height=server.UPLOAD_SIZE[1], camera_width=1280, runs=5)
def probe(source: str, upload_width: int, upload_height: int, camera_width: int, runs: int) -> None:
    """Print the time of each step of the frame path for frames of `source`, scaled to `camera_width`."""
    image = cv2.imread(source)
    if image is None:
        raise ValueError(f'{source} is not an image')
    height = round(image.shape[0] * camera_width / image.shape[1])
    frame = cv2.cvtColor(cv2.resize(image, (camera_width, height)), cv2.COLOR_BGR2RGB)
    window = np.stack([frame] * codecs.WINDOW_FRAMES)
    obs = dict.fromkeys((keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE), window)
    bound = TransformImages(upload_width, upload_height, mode=ResizeMode.BOUND)
    codec = codecs.DreamObservationCodec()
    blurrer = FaceBlurrer()

    resize_ms, bounded = _median_ms(lambda: bound.encode(obs), runs)
    jpeg_ms, packed = _median_ms(
        lambda: serialise({key: encode_jpeg(value, DEFAULT_JPEG_QUALITY) for key, value in bounded.items()}), runs
    )
    decode_ms, received = _median_ms(lambda: deserialise(packed), runs)
    cut_ms, cut = _median_ms(lambda: {key: codec._window(value, key) for key, value in received.items()}, runs)

    def scrub_all() -> None:
        for stack in cut.values():
            for view in stack:
                blurred = blurrer.scrub_frame(cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                cv2.imencode('.jpg', blurred, [cv2.IMWRITE_JPEG_QUALITY, server.JPEG_QUALITY])

    scrub_ms, _ = _median_ms(scrub_all, runs)
    sent = next(iter(bounded.values())).shape[1:3]
    print(f'Upload size {upload_width}x{upload_height}: each frame goes as {sent[1]}x{sent[0]}')
    print(f'Bytes sent: {len(packed)}')
    print(f'Rig resize: {resize_ms:.1f} ms, rig JPEG: {jpeg_ms:.1f} ms')
    print(f'Server decode: {decode_ms:.1f} ms, cut to {codecs.FRAME_SIZE}: {cut_ms:.1f} ms')
    print(f'Server face blur and JPEG of every frame: {scrub_ms:.1f} ms')


if __name__ == '__main__':
    cfn.cli(probe)
