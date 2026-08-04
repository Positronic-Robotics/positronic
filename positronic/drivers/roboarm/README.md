# Robot arms: the frame contract

Every arm driver here reports one end-effector pose (`robot_state.ee_pose`) and accepts Cartesian commands
against it. Which point that is, is a choice; this file states the choice and what a new embodiment owes.

## `default`

**Every embodiment's model declares a frame named `default`, reports `robot_state.ee_pose` in it, and accepts
Cartesian commands in it.** The model is the `urdf` the driver publishes in its `robot_meta`; `control_frame`
names the frame within it, and under this contract that name is always `default` (`models.DEFAULT_FRAME`).

Every frame conversion is measured from `default`. A policy trained in some other end-effector frame carries
the constant transform from `default` to it rather than naming it (`ChangeEEFrame`, see
[docs/codecs.md](../../../docs/codecs.md#end-effector-frames)) — which is what lets one checkpoint run on any
embodiment honouring the contract.

The harness checks the declaration on every observation, not once per episode: a remote env publishes its
`robot_meta` a turn after the reset that produced it, so at episode start there is no model to check.

## Where to put it

**Put `default` at the flange when the gripper can be swapped.** Recorded data outlives grippers, and nothing
in a dataset says where a gripper frame was. Inside the gripper is fine for an arm whose end effector is
fixed, or whose vendor frame is already the convention everybody uses.

A checkpoint's transform is portable only while every embodiment puts `default` at the same physical place.
Today's placements do not — the FR3 and the sim panda sit 45 mm apart along the approach axis — which is what
[#550](https://github.com/Positronic-Robotics/positronic/issues/550) closes.

Two things are pinned to the Franka's FR3's current placement: recordings predating this contract state a frame name
rather than an `ee_frame` transform and `ik.pose_anchor` solves at that name, and `models.DROID_EE_FRAME` is
measured from where `default` sits. `test_default_frame_still_coincides_with_the_franka_tool_frame` fails when
the placement moves.

| Embodiment | `default` sits at | Note |
| --- | --- | --- |
| Franka FR3 (`franka.py`) | `end_effector` | The arm's `F_T_EE`, 103.4 mm and −45° off the flange. Moving it to the flange is [#550](https://github.com/Positronic-Robotics/positronic/issues/550). |
| MuJoCo panda (`models.py`) | `end_effector` | The sim's grasp site. |
| RoboLab (`simulator/robolab/`) | `end_effector` | Ships the FR3 model unchanged. The env measures and drives at `droid_eef`; `RobolabAdapter` converts, and `robolab/validate.py` checks that conversion against RoboLab's own scene. |
| YAM (`yam.py`) | the `default` site in `yam.xml` | Coincident with the vendor's `grasp_site`. |
| SO-101 (`so101/driver.py`) | `gripper_frame_link` | |

## Adding an embodiment

1. Ship a model: `robot_meta` carries `urdf`, `joint_names`, `control_frame`, and the meshes the viewer needs.
2. Declare `default` in that URDF — `models.add_default_frame` attaches it to a link you name.
3. Set `control_frame` to `models.DEFAULT_FRAME`.
4. Report `robot_state.ee_pose` at that frame and drive `CartesianPosition` to it.
5. For `CartesianDelta`, compose through `CartesianDelta.apply`, which applies the command's own `frame`
   before anchoring on the measured pose.

A remote benchmark env that measures somewhere else is wrapped, not excused: give `WireCommandAdapter` an
`env_control_frame` — the transform from `default` to what the env uses — and it converts both paths
(`RobolabAdapter`).

An arm mounted away from the world origin reports poses in its own base frame, and the model carries no base
pose, so nothing downstream can reconcile the two. Keep the report and the URDF in the same frame.

# TODO: Describe what else is needed to add new robot's driver
