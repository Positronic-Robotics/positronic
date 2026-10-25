# Robot arms: the frame contract

Every arm driver here reports one end-effector pose (`robot_state.ee_pose`) and accepts Cartesian commands
against it. Which point that is, is a choice; this file states the choice and what a new embodiment owes.

## `gripper`

**Every Franka embodiment's model declares a site named `gripper` (`models.GRIPPER_SITE`), reports
`robot_state.ee_pose` at it, and accepts Cartesian commands at it.** The model is the `urdf` the driver
publishes in its `robot_meta`. `gripper` is the grip point, with the axes of the flange `link8`, so z is the
approach axis. On the FR3 with the 2F-85 it is 155 mm along that axis, where the closed pads meet. On the MuJoCo
panda it is 103.65 mm, the midpoint of the fingertip pads.

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
policy's `ee_frames` with the rig's. A remote policy gets its `ee_frames` from its server when its session opens, so
the harness compares them then, before the first observation goes to the server. A policy observes only the poses it
states at the rig's site. A policy that names another site is refused. A policy that states none observes each pose at
the rig's legacy point, the point the rig reported before `gripper`, and its pose commands go back from that point. The
legacy point is `end_effector` (`models.EE_LINK`) on a model that has it, as the FR3 and the MuJoCo panda do, and the
rig site on any other model, as YAM at `models.YAM_SITE`. The harness reads it from the rig's model and logs a warning
for each such session. This fallback is transitional (#550).

`Impedance.frame` is the stiffness point, measured from the rig site. The Franka driver accepts an offset only, and
refuses a turned frame. `DROID_IMPEDANCE` puts it at the flange.

## Old recordings

Recordings made before `gripper` state their poses at `default`, which every Franka model put at `end_effector`,
or before `ee_frames` at the site that `control_frame` names, with a codec offset in `ee_frame`.
`cfg.ds.internal.REAL_ROBOT_TRANSFORM` and `SIM_ROBOT_TRANSFORM` relabel these as `ee_frames`, and then re-express
every pose signal at `gripper` from the model the recording carries (`ik.MoveToSite`). A model with no `gripper`
gets one at the 2F-85 grip point.

Checkpoints trained on those recordings speak the old frame. They state it as an offset from `gripper`:
`models.FRANKA_HAND_FRAME` (the FR3's `F_T_EE`, 51.6 mm short of `gripper` and −45°) and `models.PANDA_HAND_FRAME` (the
sim panda's `end_effector`, 45.25 mm short of `gripper` and −45°). `test_ik.py` pins each constant to the model.

| Embodiment | Reports at | Note |
| --- | --- | --- |
| Franka FR3 (`franka.py`) | `gripper` | The driver publishes the flange pose times the 2F-85 grip offset, and solves IK at the flange. The `F_T_EE` setting in Desk changes no pose. |
| MuJoCo panda (`simulator/mujoco/sim.py`) | `gripper` | The site in `panda.xml` and `franka_table.xml`. The sim adds it to an older scene that lacks it. |
| RoboLab (`simulator/robolab/`) | `gripper` | Ships the FR3 model unchanged. The env measures and drives at `robolab_eef`; `RobolabAdapter` converts, and `robolab/validate.py` checks that conversion against RoboLab's own scene. |
| MolmoSpaces (`simulator/molmo_spaces/`) | `gripper` | Ships the FR3 model unchanged. The env measures and drives at the 2F-85 grasp site, in the arm base `fr3_link0`. MolmoSpaces' own robot base is at the bottom of the stand below it. |
| LIBERO (`simulator/libero/`) | `gripper` | `LiberoAdapter` takes the env's grip site to be the panda's `end_effector` (`PANDA_HAND_FRAME`); it is not ([#557](https://github.com/Positronic-Robotics/positronic/issues/557)). |
| YAM (`yam/driver.py`) | the `default` site in `yam.xml` (`models.YAM_SITE`) | Coincident with the vendor's `grasp_site`. It moves to `gripper` with a published YAM model. |
| SO-101 (`so101/driver.py`) | `gripper` | Coincident with `gripper_frame_link`. |

## Adding an embodiment

1. Ship a model: `robot_meta` carries `urdf`, `joint_names`, and the meshes the viewer needs.
2. Declare `gripper` in that URDF at the grip point, with the flange axes — `models.attach_frame` adds it.
3. State `ee_frames` at `models.GRIPPER_SITE` for each pose signal in the embodiment's static meta.
   `eval.ROBOT_STATIC_META` does this for one arm.
4. Report `robot_state.ee_pose` at that site and drive `CartesianPosition` to it.
5. For `CartesianDelta`, compose through `CartesianDelta.apply`, which applies the command's own `frame`
   before anchoring on the measured pose.

A remote benchmark env that measures somewhere else is wrapped, not excused: give `WireCommandAdapter` an
`env_control_frame` — the transform from `gripper` to what the env uses — and it converts both paths
(`RobolabAdapter`).

An arm mounted away from the world origin reports poses in its own base frame, and the model carries no base
pose, so nothing downstream can reconcile the two. Keep the report and the URDF in the same frame.

# TODO: Describe what else is needed to add new robot's driver
