import placo
import numpy as np

from positronic import geom


class Kinematics:
    def __init__(self, urdf_path: str, target_frame_name: str, joint_names: list[str] = None):
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.target_frame_name = target_frame_name
        self.joint_names = joint_names if joint_names is not None else list(self.robot.joint_names())
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward(self, joint_positions: np.ndarray) -> geom.Transform3D:
        for name, pos in zip(self.joint_names, joint_positions):
            self.robot.set_joint(name, pos)
        self.robot.update_kinematics()
        frame = self.robot.get_T_world_frame(self.target_frame_name)
        frame_pose = geom.Transform3D.from_matrix(frame)
        return frame_pose

    def inverse(
        self,
        current_joint_pos: np.ndarray,
        target_ee_pose: geom.Transform3D,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        n_iter: int = 10,
    ) -> np.ndarray:
        target_pose_mtx = target_ee_pose.as_matrix
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_pos[i])
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)
        self.tip_frame.T_world_frame = target_pose_mtx
        self.robot.update_kinematics()

        # For some reason, the solver doesn't converge without this loop
        for i in range(n_iter):
            self.solver.solve(True)
            self.robot.update_kinematics()

        q = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            q.append(joint)

        return np.array(q)

    @property
    def joint_limits(self) -> np.ndarray:
        return np.array([self.robot.get_joint_limits(joint_name) for joint_name in self.joint_names])
