# The bridge injects the CapApi functions and types into a program; each line that uses one waives F821.
# A fixed two-arm CapApi program: lift each hand, work both grippers, return both hands together, and go home.
# It reads no camera, so it suits a smoke run of the bridge on any bimanual scene.
state = get_robot_state()  # noqa: F821
starts = {arm: state.end_effector_poses[arm] for arm in state.arms}
report = {}

for arm, start in starts.items():
    above = [float(start.position[0]), float(start.position[1]), float(start.position[2]) + 0.05]
    target = Pose(position=above, quaternion_wxyz=start.quaternion_wxyz, frame=start.frame)  # noqa: F821
    report['lift_' + arm] = move_to_pose(target, 0.02, arm=arm)  # noqa: F821
report['close'] = set_grippers(dict.fromkeys(starts, 0.0))  # noqa: F821
report['open'] = set_grippers(dict.fromkeys(starts, 1.0))  # noqa: F821
report['back'] = move_synchronized(starts, tolerance=0.02)  # noqa: F821
for arm in starts:
    report['home_' + arm] = go_home(arm=arm)  # noqa: F821

summary = {name: (outcome.ok, outcome.diagnostics.get('move_end')) for name, outcome in report.items()}
print(summary)
failed = [name for name, outcome in report.items() if not outcome.ok]
if failed:
    raise ValueError('these calls failed: ' + str(failed) + ' ' + str(summary))
result = summary
