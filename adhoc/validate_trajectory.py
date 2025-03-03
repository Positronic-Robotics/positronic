import glob
import os

import torch
import fire
import rerun as rr

import geom


def plot_trajectories(trajectory_dir: str):
    rr.init("trajectory", spawn=True)

    for i, file in enumerate(glob.glob(os.path.join(trajectory_dir, '*.pt'))):
        data = torch.load(file)

        position = geom.Transform3D()

        for idx, (trans, rot) in enumerate(zip(data['robot_position_translation'], data['robot_position_quaternion'])):
            # Create transform from the relative transform data
            diff = geom.Transform3D(translation=trans.numpy(), quaternion=rot.numpy())

            # Apply the relative transform according to how UmiCS calculates it
            # In UmiCS: relative_gripper_transform = self.relative_gripper_transform.inv * (previous.inv * current) * self.relative_gripper_transform
            # For our trajectory validation, we're directly accumulating the relative transforms
            position = position * diff
            rr.set_time_sequence("trajectory", idx)
            rr.log(f"trajectory/{idx}", rr.Transform3D(translation=position.translation, quaternion=position.quaternion))


if __name__ == "__main__":
    fire.Fire(plot_trajectories)