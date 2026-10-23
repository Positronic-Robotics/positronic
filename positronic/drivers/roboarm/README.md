# Robot arms: the frame contract

Every arm driver here reports one end-effector pose (`robot_state.ee_pose`) and accepts Cartesian commands
against it. Which point that is, is a choice; this file states the choice and what a new embodiment owes.

## `default`

**Every embodiment's model declares a site named `default` (`models.DEFAULT_FRAME`), reports
`robot_state.ee_pose` at it, and accepts Cartesian commands at it.** The model is the `urdf` the driver
publishes in its `robot_meta`.

**Every rig, recording and pose-reading policy states where each pose signal sits, as `ee_frames`.** The entry is
`{signal: {site, offset}}` (`models.EEFrame`): the pose is `offset` from `site`, a site of the model. The
embodiment states it in its static meta for each entry of `pose_signals`, so a recording carries it.
`ik.ee_frames` reads it, and refuses a pose signal with no entry and a site that a carried model does not have.
An arm that publishes no model, such as Kinova, states no frame for any pose signal and no `pose_signals` in its
`robot_meta` (`eval.NO_MODEL_ROBOT_META`). Its rig runs joint-only policies only: the harness refuses a policy that
states frames there.

A policy trained in some other end-effector frame carries the constant transform from the rig site to it
(`ChangeEEFrame`, see [codec guide](../../policy/codecs/README.md#end-effector-frames)). The transform lets one
checkpoint run on any embodiment that keeps the contract. At the start of each episode the harness compares the
policy's `ee_frames` with the rig's. A policy observes only the poses it states at the rig's site. A policy that
states none speaks joints only and observes no pose. A policy that names another site is refused.

Recordings made before `ee_frames` state the site as `control_frame` and a codec offset as `ee_frame`.
`cfg.ds.internal.REAL_ROBOT_TRANSFORM` and `SIM_ROBOT_TRANSFORM` relabel both as `ee_frames`.

## Where to put it

**Put `default` at the flange when the gripper can be swapped.** Recorded data outlives grippers, and nothing
in a dataset says where a gripper frame was. Inside the gripper is fine for an arm whose end effector is
fixed, or whose vendor frame is already the convention everybody uses.

A checkpoint's transform is portable only while every embodiment puts `default` at the same physical place.
Today's placements do not — the FR3 and the sim panda sit 45 mm apart along the approach axis — which is what
[#550](https://github.com/Positronic-Robotics/positronic/issues/550) closes.

Recordings before #485 state their site as `end_effector`, which coincides with the FR3's `default`.
`models.DROID_EE_FRAME` is measured from that `default`. `test_default_frame_still_coincides_with_the_franka_tool_frame`
fails when the placement moves.

| Embodiment | `default` sits at | Note |
| --- | --- | --- |
| Franka FR3 (`franka.py`) | `end_effector` | The arm's `F_T_EE`, 103.4 mm and −45° off the flange. Moving it to the flange is [#550](https://github.com/Positronic-Robotics/positronic/issues/550). |
| MuJoCo panda (`models.py`) | `end_effector` | The sim's grasp site, 58.4 mm and −45° off the flange (`link8`). |
| RoboLab (`simulator/robolab/`) | `end_effector` | Ships the FR3 model unchanged. The env measures and drives at `robolab_eef`; `RobolabAdapter` converts, and `robolab/validate.py` checks that conversion against RoboLab's own scene. |
| MolmoSpaces (`simulator/molmo_spaces/`) | `end_effector` | Ships the FR3 model unchanged. The env measures and drives at the 2F-85 grasp site, in the arm base `fr3_link0`; `MolmoAdapter` converts. MolmoSpaces' own robot base is at the bottom of the stand below it. |
| YAM (`yam/driver.py`) | the `default` site in `yam.xml` | Coincident with the vendor's `grasp_site`. |
| SO-101 (`so101/driver.py`) | `gripper_frame_link` | |

Every bundled Franka model names its grip point `gripper` (`models.GRIPPER_SITE`). The frame has the axes of the
flange `link8`, so z is the approach axis. On the FR3 with the 2F-85, `gripper` is 155 mm along that axis, where
the closed pads meet. On the MuJoCo panda it is 103.65 mm, the midpoint of the fingertip pads. No embodiment reports
or solves at `gripper`.

## Adding an embodiment

1. Ship a model: `robot_meta` carries `urdf`, `joint_names`, and the meshes the viewer needs.
2. Declare `default` in that URDF — `models.add_default_frame` attaches it to a link you name.
3. State `ee_frames` at `models.DEFAULT_FRAME` for each pose signal in the embodiment's static meta.
   `eval.ROBOT_STATIC_META` does this for one arm.
4. Report `robot_state.ee_pose` at that site and drive `CartesianPosition` to it.
5. For `CartesianDelta`, compose through `CartesianDelta.apply`, which applies the command's own `frame`
   before anchoring on the measured pose.

A remote benchmark env that measures somewhere else is wrapped, not excused: give `WireCommandAdapter` an
`env_control_frame` — the transform from `default` to what the env uses — and it converts both paths
(`RobolabAdapter`).

An arm mounted away from the world origin reports poses in its own base frame, and the model carries no base
pose, so nothing downstream can reconcile the two. Keep the report and the URDF in the same frame.

# TODO: Describe what else is needed to add new robot's driver
