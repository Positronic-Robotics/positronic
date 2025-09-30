from collections.abc import Iterator

import depthai as dai

import pimm


# TODO: make this configurable
class LuxonisCamera(pimm.ControlSystem):
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.frame: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

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

                image = frame.getCvFrame()
                res = {'image': image[..., ::-1]}
                ts = frame.getTimestamp().total_seconds()

                self.frame.emit(res, ts=ts)
                yield pimm.Sleep(0.001)
