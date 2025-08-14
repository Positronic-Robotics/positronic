import configuronic as cfn

from positronic.simulator.mujoco.transforms import AddBox, AddCameras, SetBodyPosition

stack_cubes_loaders = [
    cfn.Config(AddCameras, additional_cameras={
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
    cfn.Config(AddBox, name='box_0', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[1, 0, 0, 1], freejoint=True),
    cfn.Config(SetBodyPosition, body_name='box_0_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
    cfn.Config(AddBox, name='box_1', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[0, 1, 0, 1], freejoint=True),
    cfn.Config(SetBodyPosition, body_name='box_1_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
]
