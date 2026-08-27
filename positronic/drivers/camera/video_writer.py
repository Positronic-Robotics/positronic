"""Frames off a camera port, encoded into a file. What a driver's smoke run has to show for itself."""

import logging
from collections.abc import Iterator

import av

import pimm

logger = logging.getLogger(__name__)


class VideoWriter(pimm.ControlSystem):
    """Encode whatever arrives on ``frame`` into ``filename``, at a fixed rate.

    Frames are muxed as they arrive, so a run that ends early still leaves a playable file. The rate is the
    one written into the container, not one this holds the camera to: a camera running slower than ``fps``
    yields a file that plays fast.
    """

    def __init__(self, filename: str, fps: int, codec: str = 'libx264'):
        self.filename = filename
        self.fps = fps
        self.codec = codec
        self.frame = pimm.ControlSystemReceiver[pimm.shared_memory.NumpySMAdapter](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        logger.info(f'Writing to {self.filename}')
        fps_counter = pimm.utils.RateCounter('VideoWriter')
        with av.open(self.filename, mode='w', format='mp4') as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            if not isinstance(stream, av.VideoStream):
                raise ValueError(f'{self.codec} encodes no video')
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '27', 'g': '2', 'preset': 'ultrafast', 'tune': 'zerolatency'}

            while not should_stop.value:
                frame_msg = pimm.read_updated(self.frame)
                if frame_msg is None:
                    yield pimm.Sleep(0.5 / self.fps)
                    continue

                frame = av.VideoFrame.from_ndarray(frame_msg.data.array, format='rgb24')
                container.mux(stream.encode(frame))
                fps_counter.tick()

            container.mux(stream.encode())  # frames the encoder is still holding, which no packet carries yet
