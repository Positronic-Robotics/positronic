# Trossen WidowX AI Description (MJCF)

Robot description (MJCF) of the Trossen WidowX AI arm, from
[trossen_arm_mujoco](https://github.com/TrossenRobotics/trossen_arm_mujoco)
(`trossen_arm_mujoco/assets/wxai/wxai_follower.xml` at revision
[`8d9389b`](https://github.com/TrossenRobotics/trossen_arm_mujoco/commit/8d9389b46ba02bdbf6d5e086e09d20717ccfed68)),
BSD 3-Clause, see [LICENSE](./LICENSE).

`meshdir` names `assets`, the directory beside this file that holds the meshes.

`ee_site` is the frame the arm controller reports its Cartesian position in: the site sits 0.156 m along
the flange's x axis, which is the `t_flange_tool` offset the `trossen_arm` SDK carries for the standard
`wxai_v0` end effector. Forward kinematics on this model and the pose the controller reports agree to
0.13 mm and 0.01 degrees, measured on firmware 1.11.1.
