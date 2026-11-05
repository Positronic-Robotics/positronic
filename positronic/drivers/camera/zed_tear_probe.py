"""Count the torn frames of a ZED stereo camera: frames the driver drops, and torn frames that pass its check.

Usage
  uv run --locked --extra hardware python -m positronic.drivers.camera.zed_tear_probe --serial <serial> --frames 3600
  ... --out <dir>   # also save each torn frame that passes the check, side by side, as a PNG

Point the camera at a static scene. A torn frame that passes the check then shows as a single bad frame: it differs
from both neighbours, and the neighbours match each other. The driver drops a frame when ``starts_with_zero_fill``
finds it, and the probe applies that check to the same view.
"""

import argparse
import contextlib
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from positronic.drivers.camera.zed import ZERO_FILL_BGR, sl, starts_with_zero_fill

# Mean absolute grey difference, on a 0-255 scale, above which two frames of a static scene differ.
FRAMES_DIFFER = 6.0


@dataclasses.dataclass
class TearCount:
    """What one probe run saw. Frame lists hold indexes of grabbed frames."""

    serial: int
    grabbed: int = 0
    grab_codes: dict[str, int] = dataclasses.field(default_factory=dict)
    dropped: list[int] = dataclasses.field(default_factory=list)
    torn_passed: list[int] = dataclasses.field(default_factory=list)  # single bad frames that the check sent
    partial_fill_passed: list[int] = dataclasses.field(default_factory=list)  # zero fill at row 0 start, not all of it


@dataclasses.dataclass(frozen=True)
class Sample:
    index: int
    dropped: bool
    grey: np.ndarray
    side_by_side: np.ndarray


def _small_grey(side_by_side: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2GRAY), (320, 90), interpolation=cv2.INTER_AREA).astype(
        np.int16
    )


def _sample(camera, left, right, count: TearCount) -> Sample:
    """Read one grabbed frame, apply the driver's check to it, and count the result."""
    for image, view in ((left, sl.VIEW.LEFT_UNRECTIFIED), (right, sl.VIEW.RIGHT_UNRECTIFIED)):
        result = camera.retrieve_image(image, view)
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f'Camera {count.serial} grabbed frame {count.grabbed} and gave no {view} image: {result}'
            )
    raw_left = left.get_data()
    dropped = starts_with_zero_fill(raw_left)
    if dropped:
        count.dropped.append(count.grabbed)
    elif np.all(raw_left[0, 0, :3] == ZERO_FILL_BGR):
        count.partial_fill_passed.append(count.grabbed)
    side_by_side = np.concatenate([raw_left[:, :, :3], right.get_data()[:, :, :3]], axis=1)
    return Sample(count.grabbed, dropped, _small_grey(side_by_side), side_by_side)


@contextlib.contextmanager
def _opened(serial: int, resolution, fps: int) -> Iterator:
    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = resolution
    init.camera_fps = fps
    init.set_from_serial_number(serial)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.async_grab_camera_recovery = True
    result = camera.open(init)
    if result != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f'Failed to open camera {serial}: {result}')
    try:
        yield camera
    finally:
        camera.close()


def _single_bad_frame(before: np.ndarray, frame: np.ndarray, after: np.ndarray) -> bool:
    d_before, d_after = np.abs(frame - before).mean(), np.abs(after - frame).mean()
    return (
        d_before > FRAMES_DIFFER
        and d_after > FRAMES_DIFFER
        and np.abs(after - before).mean() < 0.5 * min(d_before, d_after)
    )


def probe(serial: int, frames: int, resolution, fps: int, out: Path | None) -> TearCount:
    """Grab ``frames`` frames at ``resolution`` (an ``sl.RESOLUTION``) and count the torn ones."""
    left, right = sl.Mat(), sl.Mat()
    count = TearCount(serial)
    window: list[Sample] = []
    with _opened(serial, resolution, fps) as camera:
        for _ in range(frames):
            result = camera.grab()
            count.grab_codes[str(result)] = count.grab_codes.get(str(result), 0) + 1
            if result != sl.ERROR_CODE.SUCCESS:
                continue
            window = [*window[-2:], _sample(camera, left, right, count)]
            count.grabbed += 1
            if len(window) < 3 or window[1].dropped or not _single_bad_frame(*(w.grey for w in window)):
                continue
            count.torn_passed.append(window[1].index)
            path = out / f'{serial}_passed_{window[1].index}.png' if out is not None else None
            if path is not None and not cv2.imwrite(str(path), window[1].side_by_side):
                raise OSError(f'Failed to write {path}')
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--serial', type=int, required=True)
    parser.add_argument('--frames', type=int, default=3600)
    parser.add_argument('--resolution', default='HD720', type=str.upper, choices=list(sl.RESOLUTION.__members__))
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
    count = probe(args.serial, args.frames, sl.RESOLUTION[args.resolution], args.fps, args.out)
    print(json.dumps(dataclasses.asdict(count), indent=1))


if __name__ == '__main__':
    main()
