import av
import ironic as ir


@ir.ironic_system(
    input_ports=['image']
)
class VideoDumper(ir.ControlSystem):
    def __init__(self, filename: str, fps: int, width: int = None, height: int = None, codec: str = 'libx264'):
        super().__init__()
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.container = None
        self.stream = None
        self.fps_counter = ir.utils.FPSCounter("VideoDumper")

    async def setup(self):
        self.container = av.open(self.filename, mode='w', format='mp4')
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {
            'crf': '27',
            'g': '2',
            'preset': 'ultrafast',
            'tune': 'zerolatency'
        }

    async def cleanup(self):
        if self.stream:
            packet = self.stream.encode(None)
            self.container.mux(packet)
        if self.container:
            self.container.close()

    @ir.on_message('image')
    async def handle_image(self, message: ir.Message):
        if message.data is None:
            return

        if self.stream.width is None:  # First frame
            self.stream.width = self.width or message.data.shape[1]
            self.stream.height = self.height or message.data.shape[0]

        frame = av.VideoFrame.from_ndarray(message.data, format='bgr24')
        packet = self.stream.encode(frame)
        self.container.mux(packet)
        self.fps_counter.tick()
