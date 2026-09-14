# Trossen WidowX AI Description

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

`wxai_follower.urdf` is the same arm as URDF, from
[trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)
(`urdf/generated/wxai/wxai_follower.urdf`), BSD 3-Clause, with `meshes/trossen_black.png` beside the STLs
it already shares with the MJCF. The driver solves against the MJCF and the codecs against the URDF, so
`ee_gripper_link` of the URDF and `ee_site` of the MJCF must name one place:
`test_the_trossen_urdf_and_its_mjcf_put_the_control_frame_in_the_same_place` measures that they do.

The URDF names its meshes the way its own ROS package does; the driver shortens each to the file beside
this README before it publishes the model.
