from typing import Iterator
import ironic as ir
import depthai as dai


# TODO: make this configurable
class LuxonisCamera:
    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, fps: int = 60):
        super().__init__()
        self.fps = fps

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock) -> Iterator[ir.Sleep]:
        pipeline = dai.Pipeline()
        pipeline.setXLinkChunkSize(0)  # increases speed

        camColor = pipeline.create(dai.node.ColorCamera)
        camColor.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camColor.setFps(self.fps)
        camColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camColor.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        xoutColor = pipeline.create(dai.node.XLinkOut)
        xoutColor.setStreamName("image")

        camColor.isp.link(xoutColor.input)
        fps_counter = ir.utils.RateCounter('luxonis')

        with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
            queue = device.getOutputQueue("image", 8, blocking=False)
            while not should_stop.value:
                frame = queue.tryGet()
                if frame is None:
                    yield ir.Sleep(0.001)
                    continue
                fps_counter.tick()

                image = frame.getCvFrame()
                res = {'image': image[..., ::-1]}
                ts = frame.getTimestamp().total_seconds()

                self.frame.emit(res, ts=ts)
                yield ir.Sleep(0.001)
