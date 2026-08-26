# Trossen WidowX AI Description (MJCF)

Robot description (MJCF) of the Trossen WidowX AI arm, from
[trossen_arm_mujoco](https://github.com/TrossenRobotics/trossen_arm_mujoco)
(`trossen_arm_mujoco/assets/wxai/wxai_follower.xml`), BSD 3-Clause, see [LICENSE](./LICENSE).

Vendored whole apart from one change: `meshdir` names `assets`, where the meshes it uses now sit.

`scene.xml` is not from upstream. It puts the arm on a lit ground plane, for a viewer to open: upstream's
own scene wraps `wxai_base.xml`, which is a different arm.

`ee_site` is the frame the arm controller reports its Cartesian position in: the site sits 0.156 m along
the flange's x axis, which is the `t_flange_tool` offset the `trossen_arm` SDK carries for the standard
`wxai_v0` end effector. Forward kinematics on this model and the pose the controller reports agree to
0.13 mm and 0.01 degrees, measured on firmware 1.11.1.
