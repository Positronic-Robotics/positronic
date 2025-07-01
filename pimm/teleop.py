from pimm.gui.gui import DearpyguiUi
from pimm.simulator.mujoco.sim import MujocoSim
from pimm.drivers.webxr import WebXR

import ironic2 as ir


with ir.World() as world:
    gui = DearpyguiUi()
    webxr = WebXR(port=8000)
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

    world.start(sim.run)
    gui.init()
    gui.run()