# Robot arms: the frame contract

Every arm driver here reports one end-effector pose (`robot_state.ee_pose`) and accepts Cartesian commands
against it. Which point that is, is a choice; this file states the choice and what a new embodiment owes.

## `default`

**Every embodiment's model declares a frame named `default`, reports `robot_state.ee_pose` in it, and accepts
Cartesian commands in it.** The model is the `urdf` the driver publishes in its `robot_meta`; `control_frame`
names the frame within it, and under this contract that name is always `default` (`models.DEFAULT_FRAME`).

Every frame conversion is measured from `default`. A policy trained in some other end-effector frame carries
the constant transform from `default` to it rather than naming it (`ChangeEEFrame`, see
[codec guide](../../policy/codecs/README.md#end-effector-frames)) — which is what lets one checkpoint run on any
embodiment honouring the contract.

The harness checks the declaration on every observation, not once per episode: a remote env publishes its
`robot_meta` a turn after the reset that produced it, so at episode start there is no model to check.

## Where to put it

**Put `default` at the flange when the gripper can be swapped.** Recorded data outlives grippers, and nothing
in a dataset says where a gripper frame was. Inside the gripper is fine for an arm whose end effector is
fixed, or whose vendor frame is already the convention everybody uses.

A checkpoint's transform is portable only while every embodiment puts `default` at the same physical place.
Every Franka model puts it at the flange, `link8` (`models.FLANGE_LINK`), so one transform means the same point
on the real arm and in each simulator.

A recording or a checkpoint made while `default` sat at the Franka tool frame speaks poses there. The two old
points are constants from the flange: `models.FR3_EE_FRAME` for the real arm and `models.PANDA_EE_FRAME` for the
sim panda. `cfg.ds.internal`'s robot transforms read an old recording at the flange, from the model it carries
(`ik.MoveDefaultFrame`), and a checkpoint trained before the move states its old frame with `ee_frame`.

| Embodiment | `default` sits at | Note |
| --- | --- | --- |
| Franka FR3 (`franka.py`) | `link8`, the flange | positronic-franka publishes the flange pose and solves the IK there. The Desk tool frame `F_T_EE` does not move it. |
| MuJoCo panda (`models.py`) | `link8`, the flange | The sim reports and solves at its `link8` site. |
| RoboLab (`simulator/robolab/`) | `link8`, the flange | Ships the FR3 model. The env measures and drives at `droid_eef`; `RobolabAdapter` converts with `models.DROID_EE_FRAME`, and `robolab/validate.py` checks that conversion against RoboLab's own scene. |
| MolmoSpaces (`simulator/molmo_spaces/`) | `link8`, the flange | Ships the FR3 model. The env measures and drives at the 2F-85 grasp site, in the arm base `fr3_link0`; `MolmoAdapter` converts. MolmoSpaces' own robot base is at the bottom of the stand below it. |
| LIBERO (`simulator/libero/`) | `link8`, the flange | Ships the panda model. `LiberoAdapter` converts with `models.PANDA_EE_FRAME`, the old panda point, which is 38.1 mm and 90° off the eef site LIBERO drives ([#557](https://github.com/Positronic-Robotics/positronic/issues/557)). |
| YAM (`yam/driver.py`) | the `default` site in `yam.xml` | Coincident with the vendor's `grasp_site`. |
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
