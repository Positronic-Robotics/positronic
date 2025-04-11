import ironic as ir

from positronic.simulator.mujoco.scene.transforms import AddBox, AddCameras, SetBodyPosition
from positronic.simulator.mujoco.sim import create_from_config


def _list(*args):
    return list(args)

stack_cubes_loaders = ir.Config(
    _list,
    ir.Config(AddCameras, additional_cameras={
        'side_view': {
            'pos': [1.235, -0.839, 1.092],
            'xyaxes': [0.712, 0.702, -0.000, -0.420, 0.425, 0.802]
        },
        'table_view': {
            'pos': [0.985, -0.008, 0.744],
            'xyaxes': [0.003, 1.000, 0.000, -0.855, 0.003, 0.518]
        },
        'front_view': {
            'pos': [1.756, 0.061, 0.850],
            'xyaxes': [-0.009, 1.000, 0.000, -0.328, -0.003, 0.945]
        },
        'back_view': {
            'pos': [-0.451, 0.978, 0.629],
            'xyaxes': [-0.544, -0.839, -0.000, 0.242, -0.157, 0.958]
        },
    }),
    ir.Config(AddBox, name='box_0', size=[0.025, 0.025, 0.025], pos=[0.4, 0.0, 0.31], rgba=[1, 0, 0, 1], freejoint=True),
    ir.Config(SetBodyPosition, body_name='box_0_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
    ir.Config(AddBox, name='box_1', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[0, 1, 0, 1], freejoint=True),
    ir.Config(SetBodyPosition, body_name='box_1_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
)


simulator = ir.Config(
    create_from_config,
    mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
    camera_names=['handcam_back', 'handcam_front', 'front_view', 'back_view', 'side_view'],
    camera_width=1280,
    camera_height=720,
    simulation_hz=500,
    loaders=stack_cubes_loaders
)

simulator_droid = simulator.override(
    camera_names=['handcam_left_back', 'side_view'],
)