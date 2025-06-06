import abc
import ironic2 as ir
from typing import Dict, List


class CameraSystem(abc.ABC):

    @abc.abstractmethod
    def run(self, should_stop: ir.SignalReader):
        pass

    @property
    @abc.abstractmethod
    def frame(self) -> ir.SignalReader:
        pass

    @frame.setter
    @abc.abstractmethod
    def frame(self, frame: ir.SignalReader):
        pass


class MergedCamera:

    frame: ir.SignalReader = ir.NoOpReader()

    def __init__(self, **cameras: Dict[str, CameraSystem]):
        self.cameras = cameras
        self.process_world = ir.mp.MPWorld()

    def _main_loop(self, should_stop: ir.SignalReader):
        frames = {}
        while not should_stop.value():
            for camera_name, camera in self.cameras.items():
                frames[camera_name] = camera.frame.value()
            self.frame.emit(ir.Message(data=frames, ts=ir.system_clock()))

    def run(self, should_stop: ir.SignalReader):
        self.process_world.run(self._main_loop, [camera.run for camera in self.cameras.values()])