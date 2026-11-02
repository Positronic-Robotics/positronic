"""Poses of the earlier bundled YAM model, which YAM recordings and checkpoints speak, as fixed references."""

# Joints, and the pose of ``GRIPPER_SITE`` at them in the arm-base frame of the earlier bundled YAM model.
EARLIER_MODEL_POSES = [
    (
        [0.5, 1.2, 1.0, -0.6, 0.4, 1.5],
        [0.319878484, 0.140564402, 0.179219946],
        [0.10141251, 0.892468315, 0.184023542, 0.399188108],
    ),
    (
        [2.6, 0.3, 3.0, 0.0, 1.0, 2.0],
        [0.635856594, -0.235739914, 0.302116541],
        [0.754621015, -0.139149565, 0.604301441, 0.214486109],
    ),
    ([0.0, 1.047, 1.047, 0.0, 0.0, 0.0], [0.380254836, 0.0, 0.351804625], [0.5, 0.5, 0.5, 0.5]),
    (
        [-0.9, 1.6, 2.2, 0.8, 0.9, -1.2],
        [0.022114321, -0.284280157, 0.702958701],
        [0.856238756, 0.394382419, -0.19539657, -0.270440159],
    ),
    (
        [1.3, 0.8, 1.9, -1.4, -0.5, 0.3],
        [-0.06546205, 0.114352508, 0.436602296],
        [-0.207494703, 0.078654489, 0.789610766, 0.572078892],
    ),
]
# Within this distance and angle, the reported ee_pose is the one the earlier model reported.
SAME_POSE_M, SAME_POSE_RAD = 1e-5, 2e-5
