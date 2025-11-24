import configuronic as cfn

from positronic.simulator.mujoco.transforms import (
    AddBox,
    AddCameras,
    AddObjectsInTote,
    AddTote,
    SetBodyPosition,
    SetTwoObjectsPositions,
)

stack_cubes_loaders = [
    cfn.Config(
        AddCameras,
        additional_cameras={
            'side_view': {'pos': [1.235, -0.839, 1.092], 'xyaxes': [0.712, 0.702, -0.000, -0.420, 0.425, 0.802]},
            'table_view': {'pos': [0.985, -0.008, 0.744], 'xyaxes': [0.003, 1.000, 0.000, -0.855, 0.003, 0.518]},
            'front_view': {'pos': [1.756, 0.061, 0.850], 'xyaxes': [-0.009, 1.000, 0.000, -0.328, -0.003, 0.945]},
            'back_view': {'pos': [-0.451, 0.978, 0.629], 'xyaxes': [-0.544, -0.839, -0.000, 0.242, -0.157, 0.958]},
        },
    ),
    cfn.Config(AddBox, name='box_0', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[1, 0, 0, 1], freejoint=True),
    cfn.Config(SetBodyPosition, body_name='box_0_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
    cfn.Config(AddBox, name='box_1', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[0, 1, 0, 1], freejoint=True),
    cfn.Config(SetBodyPosition, body_name='box_1_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
]


multi_tote_loaders = [
    cfn.Config(
        AddCameras,
        additional_cameras={
            'side_view': {'pos': [1.235, -0.839, 1.092], 'xyaxes': [0.712, 0.702, -0.000, -0.420, 0.425, 0.802]},
            'table_view': {'pos': [0.985, -0.008, 0.744], 'xyaxes': [0.003, 1.000, 0.000, -0.855, 0.003, 0.518]},
            'front_view': {'pos': [1.756, 0.061, 0.850], 'xyaxes': [-0.009, 1.000, 0.000, -0.328, -0.003, 0.945]},
            'back_view': {'pos': [-0.451, 0.978, 0.629], 'xyaxes': [-0.544, -0.839, -0.000, 0.242, -0.157, 0.958]},
        },
    ),
    cfn.Config(AddTote, name='tote_0', size=[0.08, 0.12, 0.03], pos=[0, 0, 0.3], rgba=[1, 0, 0, 1]),
    cfn.Config(AddTote, name='tote_1', size=[0.08, 0.12, 0.03], pos=[0, 0, 0.3], rgba=[0, 1, 0, 1]),
    cfn.Config(
        SetTwoObjectsPositions,
        object1_name='tote_0',
        object2_name='tote_1',
        table_bounds=((0.35, 0.65), (-0.2, 0.2)),
        min_distance=0.25,
    ),
    cfn.Config(
        AddObjectsInTote,
        tote_name='tote_0',
        object_name_prefix='obj',
        num_objects=10,
        object_size=[0.015, 0.015, 0.015],
        tote_size=[0.08, 0.12, 0.03],
        rgba=[0, 0, 1, 1],
    ),
]
