# i2rt YAM model

`yam.urdf`, `yam.xml` and the meshes come from [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) at commit
`5d47b358bafb30c65e397f2ece506550a0db4594` (release v1.2.4), under the MIT licence in `LICENSE`. The YAM driver
uses the `i2rt` package at the same commit.

- `yam.xml` is `i2rt/robot_models/arm/yam/yam.xml` and `i2rt/robot_models/gripper/linear_4310/linear_4310.xml`,
  combined as `i2rt.robots.utils.combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)` combines them.
- `yam.urdf` is `i2rt/robot_models/arm/yam/yam.urdf`.
- `base.stl` and `link1.stl` to `link5.stl` come from `i2rt/robot_models/arm/yam/assets`. `gripper.stl`,
  `tip_left.stl` and `tip_right.stl` come from `i2rt/robot_models/gripper/linear_4310/assets`.

Changes from the source:

- The two models and the meshes are in one directory. Each mesh `file` and `filename` is the file name only.
- Quadric decimation takes `link2.stl` from 29,914 and `link3.stl` from 30,150 triangles to 20,000 each. This keeps
  each file below the 1000 KB limit of the repository. The other meshes are copied without change.
- `yam.xml` has no `tcp_site` and no `grasp_site`. The `grasp_site` that YAM recordings name is a different point,
  0.3 mm away (`models.YAM_GRASP_SITE_FRAME`).
- In `yam.urdf`, the child link of `joint6` is `link6`, as in `yam.xml`. The source names it `gripper`, and that is
  the name of the grip point (`models.GRIPPER_SITE`).
- In `yam.urdf`, the links `tip_left` and `tip_right` and the joints `joint7` and `joint8` are the bodies of
  `linear_4310.xml`, written as URDF. The fingertips of the source URDF are 2.1 mm longer. Thus the URDF and the MJCF
  draw the same gripper.

## The `gripper` site

`yam.xml` has the site `gripper` in the body `gripper`. `models.bundled_yam_model()` adds the link `gripper` to the
URDF at the same pose. The site is 40.8 mm off the wrist axis, where the closed fingertips of the crank gripper of
ABC's YAM model meet. The linear gripper closes on the wrist axis.

The base frame of this model is at the bottom of the arm base, 9.5 mm below the base frame of ABC's YAM model. Each
YAM mount is 9.5 mm lower than the base frame of ABC's model, so the two models put `gripper` at the same world pose
for the same joints.
