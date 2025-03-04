import depthai as dai
import ironic as ir


# TODO: make this configurable
class LuxonisCamera:
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
        self.fps = ir.utils.FPSCounter('luxonis')

    def setup(self):
        self.device = dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
        self.device.__enter__()
        self.queue = self.device.getOutputQueue("image", 8, blocking=False)

    def get_frame(self):
        frame = self.queue.tryGet()
        if frame is None:
            return None, None
        self.fps.tick()

        image = frame.getCvFrame()
        res = {
            'image': image[..., ::-1]
        }

        ts = frame.getTimestamp().total_seconds() * 1e9

        return res, int(ts)

    def cleanup(self):
        self.device.__exit__(None, None, None)


@ir.ironic_system(output_ports=['frame'])
class LuxonisCameraCS(ir.ControlSystem):
    def __init__(self, camera: LuxonisCamera):
        super().__init__()
        self.camera = camera

    async def setup(self):
        self.camera.setup()

    async def step(self):
        frame, timestamp = self.camera.get_frame()
        if frame is not None:
            await self.outs.frame.write(ir.Message(data=frame, timestamp=timestamp))
        return ir.State.ALIVE
