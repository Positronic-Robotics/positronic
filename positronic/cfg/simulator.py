import configuronic as cfn

from positronic import keys
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.simulator.mujoco.transforms import (
    AddBox,
    AddCameras,
    AddObjectsInTote,
    AddTote,
    SetBodyPosition,
    SetRenderQuality,
    SetTwoObjectsPositions,
)
from positronic.utils import package_assets_path

# The Franka table scene's camera ports, under the canonical observation names.
# Three, not two: MuJoCo does not render the second image when only two cameras are bound.
MUJOCO_FRANKA_CAMERAS = {
    keys.WRIST_IMAGE: 'handcam_left_ph',
    keys.EXTERIOR_IMAGE: 'back_view_ph',
    keys.AGENT_VIEW_IMAGE: 'agentview',
}


@cfn.config()
def stack_cubes_loaders():
    return [
        AddCameras(
            additional_cameras={
                'side_view': {'pos': [1.235, -0.839, 1.092], 'xyaxes': [0.712, 0.702, -0.000, -0.420, 0.425, 0.802]},
                'table_view': {'pos': [0.985, -0.008, 0.744], 'xyaxes': [0.003, 1.000, 0.000, -0.855, 0.003, 0.518]},
                'front_view': {'pos': [1.756, 0.061, 0.850], 'xyaxes': [-0.009, 1.000, 0.000, -0.328, -0.003, 0.945]},
                'back_view': {'pos': [-0.451, 0.978, 0.629], 'xyaxes': [-0.544, -0.839, -0.000, 0.242, -0.157, 0.958]},
            }
        ),
        AddBox(name='box_0', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[1, 0, 0, 1], freejoint=True),
        SetBodyPosition(body_name='box_0_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
        AddBox(name='box_1', size=[0.02, 0.02, 0.01], pos=[0.0, 0.0, 0.01], rgba=[0, 1, 0, 1], freejoint=True),
        SetBodyPosition(body_name='box_1_body', random_position=[[0.31, -0.28, 0.31], [0.69, 0.28, 0.31]]),
    ]


@cfn.config(num_objects=10)
def multi_tote_loaders(num_objects: int):
    return [
        AddCameras(
            additional_cameras={
                'side_view': {'pos': [1.235, -0.839, 1.092], 'xyaxes': [0.712, 0.702, -0.000, -0.420, 0.425, 0.802]},
                'table_view': {'pos': [0.985, -0.008, 0.744], 'xyaxes': [0.003, 1.000, 0.000, -0.855, 0.003, 0.518]},
                'front_view': {'pos': [1.756, 0.061, 0.850], 'xyaxes': [-0.009, 1.000, 0.000, -0.328, -0.003, 0.945]},
                'back_view': {'pos': [-0.451, 0.978, 0.629], 'xyaxes': [-0.544, -0.839, -0.000, 0.242, -0.157, 0.958]},
            }
        ),
        AddTote(name='tote_0', size=[0.08, 0.12, 0.03], pos=[0, 0, 0.3], rgba=[1, 0, 0, 1]),
        AddTote(name='tote_1', size=[0.08, 0.12, 0.03], pos=[0, 0, 0.3], rgba=[0, 1, 0, 1]),
        SetTwoObjectsPositions(
            object1_name='tote_0',
            object2_name='tote_1',
            table_bounds=((0.35, 0.65), (-0.2, 0.2)),
            min_distance=0.25,
            object_sizes=([0.08, 0.12, 0.03], [0.08, 0.12, 0.03]),
        ),
        AddObjectsInTote(
            tote_name='tote_0',
            object_name_prefix='obj',
            num_objects=num_objects,
            object_size=[0.015, 0.015, 0.015],
            tote_size=[0.08, 0.12, 0.03],
            rgba=[0, 0, 1, 1],
        ),
    ]


@cfn.config(loaders=stack_cubes_loaders, shadowsize=0, offsamples=0, reflectance=0.0)
def low_render_quality(loaders, shadowsize: int, offsamples: int, reflectance: float):
    """``loaders`` with the render-cost knobs turned down.

    Shadows, multisampling and specular reflections are most of an offscreen frame under a software GL
    stack; dropping them renders the Franka table scene 7x faster (``SetRenderQuality``), which is what
    lets a CPU-only box keep an attended sim at wall-clock pace. Pass the bare loaders instead where
    the box has a GPU or where those effects are part of what is being evaluated.
    """
    return [*loaders, SetRenderQuality(shadowsize, offsamples, reflectance)]


# The rate a Franka rollout is driven at: one scheduler round, camera frame and inference per period.
MUJOCO_FRANKA_CONTROL_HZ = 15

mujoco_franka_sim = cfn.Config(
    MujocoSim,
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=low_render_quality,
    camera_fps=MUJOCO_FRANKA_CONTROL_HZ,
    control_period=1 / MUJOCO_FRANKA_CONTROL_HZ,
)
