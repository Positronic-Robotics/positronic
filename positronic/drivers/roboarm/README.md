# Robot arms: the frame contract

Every arm driver here reports one end-effector pose (`robot_state.ee_pose`) and accepts Cartesian commands
against it. "End-effector" is not a physical fact — it is a point somebody chose — so this file states which
point, and what a new embodiment owes the rest of the system.

## `default`

**Every embodiment's model declares a frame named `default`, reports `robot_state.ee_pose` in it, and accepts
Cartesian commands in it.** The model is the `urdf` the driver publishes in its `robot_meta`; `control_frame`
names the frame within it, and for a rig built to this contract that name is always `default`
(`models.DEFAULT_FRAME`).

`default` is the origin of the coordinate system every frame conversion is expressed in. A policy trained in
some other end-effector frame does not name that frame to the rig — it carries the constant transform from
`default` to it (`ChangeEEFrame`, see [docs/codecs.md](../../../docs/codecs.md#end-effector-frames)). That is
what lets one checkpoint run on any embodiment honouring the contract, and it is why `default` has to mean
something stable.

The harness checks the declaration at the start of every episode: a `control_frame` the model does not carry
raises before the arm moves.

## Where to put it

**Put `default` at the flange when the gripper can be swapped.** Recorded data outlives grippers. An episode
anchored to a gripper frame becomes unplaceable the moment that gripper comes off — nothing in the dataset
says where that frame was — while one anchored to the flange stays valid for the life of the arm.

Putting it inside the gripper is fine for an arm whose end effector is fixed, or one whose vendor frame is
already the convention everybody uses.

Current placements:

| Embodiment | `default` sits at | Note |
| --- | --- | --- |
| Franka FR3 (`franka.py`) | `end_effector` | The arm's `F_T_EE`, 103.4 mm and −45° off the flange. Moving it to the flange is [#550](https://github.com/Positronic-Robotics/positronic/issues/550). |
| MuJoCo panda (`models.py`) | `end_effector` | The sim's grasp site. |
| RoboLab (`simulator/robolab/`) | `end_effector` | It ships the FR3 model unchanged. The env itself measures and drives at `droid_eef`; `RobolabAdapter` converts, which is what lets one checkpoint run on RoboLab and on the real rig. |
| YAM (`yam.py`) | the `default` site in `yam.xml` | Coincident with the vendor's `grasp_site`. |
| SO-101 (`so101/driver.py`) | `gripper_frame_link` | |

## Adding an embodiment

1. Ship a model: `robot_meta` carries `urdf`, `joint_names`, `control_frame`, and the meshes the viewer needs.
2. Declare `default` in that URDF — `models.add_default_frame` attaches it to a link you name.
3. Set `control_frame` to `models.DEFAULT_FRAME`.
4. Report `robot_state.ee_pose` at that frame and drive `CartesianPosition` to it.
5. For `CartesianDelta`, compose through `command.apply_cartesian_delta`, which applies the command's own
   `frame` before anchoring on the measured pose.

A remote benchmark env that measures somewhere else is wrapped, not excused: give `WireCommandAdapter` the
constant `control_frame` from `default` to what the env uses, and it converts both paths (`RobolabAdapter`).

An arm mounted away from the world origin reports poses in its own base frame; the model carries no base pose,
so nothing downstream can reconcile the two. Keep the report and the URDF in the same frame.
