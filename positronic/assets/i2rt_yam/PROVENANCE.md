# i2rt YAM URDF

`yam.urdf` and the `*_visual.stl` meshes come from
[i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) at commit
`d4efb66d81bd8bde42909880b16591d4af82e8c0`, directory `robot_models/yam`, under the MIT licence in
`LICENSE`.

The MJCF in `assets/mujoco/i2rt_yam` comes from the same URDF, so the two models have the same
kinematics. The YAM driver solves FK and IK against that MJCF. At `YAM_SITE`, the two models agree to
within 0.01 mm and 0.0001 rad over the joint range. The i2rt URDF at commit
`120c3c81400171174604e503943f8d1ebc891058`, `i2rt/robot_models/arm/yam/v1`, has other link names and other
link offsets, so it does not match the MJCF. For example, its `joint1` is 68.0 mm above the base, not 63.1 mm.

Changes from the source:

- The `collision` elements are removed, and the mesh files they name are not copied.
- Each mesh `filename` is the file name only, without `package://assets/`.
- The limits of `joint2` and `joint6` are the MJCF ranges, which the driver enforces: `[0, 3.66519]` and
  `[-2.0944, 2.0944]`.

The meshes are copied without change.
