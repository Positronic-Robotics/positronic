from collections.abc import Iterator
from datetime import timedelta

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.camera import CAPTURE_TIME, HANDOFF_TIME

# TODO: `depthai` is in no extra, so this driver has no supported install path. Add it to `hardware` (with the
# platform markers the SDK needs) or drop the driver.
with vendor_import('depthai', 'Luxonis camera support', '`depthai` is in no extra of this project.'):
    import depthai as dai


# TODO: make this configurable
class LuxonisCamera(pimm.ControlSystem):
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.frame: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self._frame_adapter = None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        pipeline = dai.Pipeline()
        pipeline.setXLinkChunkSize(0)  # increases speed

        camColor = pipeline.create(dai.node.ColorCamera)
        camColor.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camColor.setFps(self.fps)
        camColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camColor.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        xoutColor = pipeline.create(dai.node.XLinkOut)
        xoutColor.setStreamName('image')

        camColor.isp.link(xoutColor.input)
        fps_counter = pimm.utils.RateCounter('luxonis')

        with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
            queue = device.getOutputQueue('image', 8, blocking=False)
            while not should_stop.value:
                frame = queue.tryGet()
                if frame is None:
                    yield pimm.Sleep(0.001)
                    continue
                fps_counter.tick()

                image = frame.getCvFrame()[..., ::-1]  # BGR to RGB
                capture_ns = _ns(frame.getTimestamp())

                self._frame_adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(image, self._frame_adapter)
                # The device stamps a frame on the host's steady clock, which `dai.Clock.now()` reads.
                frame_time = pimm.Time(**{CAPTURE_TIME: capture_ns, HANDOFF_TIME: _ns(dai.Clock.now())})
                self.frame.emit(self._frame_adapter, time=frame_time)
                yield pimm.Sleep(0.001)


def _ns(stamp: timedelta) -> int:
    return round(stamp.total_seconds() * 1e9)
