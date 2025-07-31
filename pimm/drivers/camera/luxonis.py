from typing import Iterator
import depthai as dai
import ironic2 as ir


# TODO: make this configurable
class LuxonisCamera:
    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, fps: int = 60):
        super().__init__()
        self.pipeline = dai.Pipeline()
        self.pipeline.setXLinkChunkSize(0)  # increases speed

        self.camColor = self.pipeline.create(dai.node.ColorCamera)
        self.camColor.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.camColor.setFps(fps)
        self.camColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camColor.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        self.xoutColor = self.pipeline.create(dai.node.XLinkOut)
        self.xoutColor.setStreamName("image")

        self.camColor.isp.link(self.xoutColor.input)
        self.fps_counter = ir.utils.FPSCounter('luxonis')
        self.rate_limiter = ir.utils.RateLimiter(fps)

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock) -> Iterator[ir.Sleep]:
        with dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
            queue = device.getOutputQueue("image", 8, blocking=False)

            while not should_stop.value:
                frame = queue.tryGet()
                if frame is None:
                    yield ir.Sleep(0.001)
                    continue
                self.fps_counter.tick()

                image = frame.getCvFrame()
                res = {'image': image[..., ::-1]}
                ts = frame.getTimestamp().total_seconds()

                self.frame.emit(res, ts=ts)
                yield self.rate_limiter.wait_time()
