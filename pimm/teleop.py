from pimm.gui.gui import DearpyguiUi
from pimm.simulator.mujoco.sim import MujocoSim
from pimm.drivers.webxr import WebXR

import ironic2 as ir
import geom

class _Tracker:
    on = False
    _offset = geom.Transform3D()
    _teleop_t = geom.Transform3D()

    def __init__(self, operator_position: geom.Transform3D | None):
        self._operator_position = operator_position
        self.on = self.umi_mode

    @property
    def umi_mode(self):
        return self._operator_position is None

    def turn_on(self, robot_pos: geom.Transform3D):
        if self.umi_mode:
            print("Ignoring tracking on/off in UMI mode")
            return

        self.on = True
        print("Starting tracking")
        self._offset = geom.Transform3D(
            -self._teleop_t.translation + robot_pos.translation,
            self._teleop_t.rotation.inv * robot_pos.rotation,
        )

    def turn_off(self):
        if self.umi_mode:
            print("Ignoring tracking on/off in UMI mode")
            return
        self.on = False
        print("Stopped tracking")

    def update(self, tracker_pos: geom.Transform3D):
        if self.umi_mode:
            return tracker_pos

        self._teleop_t = self._operator_position * tracker_pos * self._operator_position.inv
        return geom.Transform3D(
            self._teleop_t.translation + self._offset.translation,
            self._teleop_t.rotation * self._offset.rotation
        )


with ir.World() as world:
    gui = DearpyguiUi()
    webxr = WebXR(port=5005)
    sim = MujocoSim(
        mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
        simulation_hz=500,
        camera_names=['handcam_back', 'handcam_front'],
        camera_width=640,
        camera_height=480,
        loaders=()
    )

    gui.buttons, buttons_out = world.pipe()
    for cam_name in sim.camera_names:
        sim.frames[cam_name], frame_out = world.pipe()
        gui.cameras[cam_name] = frame_out

    world.start(sim.run, gui.run)

    last_ts = ir.system_clock()
    while not ir.is_true(world.should_stop):
        ts = ir.system_clock()
        delta = ts - last_ts

        sim.step(delta)