# Integrating a simulator

positronic drives a simulator as an embodiment: the same signals, commands and recordings as a physical rig.
`env_server/adapter.py` maps between the two. Some of what the adapter needs is known only to the simulator;
this file states those, and each is a claim to check against the running env.

## The end-effector frame

positronic reports `robot_state.ee_pose` at the frame its model calls `default` (see
[the frame contract](../drivers/roboarm/README.md)). A simulator measures and drives at a frame of its own.
`WireCommandAdapter` converts with one constant, `env_control_frame`, placing the env's frame relative to
`default`.

**State that constant against a body both models carry, and check it against the running env.** Use the
flange: it is a mounting face, so the hardware dictates where it sits rather than whoever authored the model.

    default -> env frame  =  (default -> flange, ours)  ∘  (flange -> env frame, the env's)

Naming a gripper frame instead assumes both projects chose the same gripper geometry.

A wrong constant cancels inside the loop — the env drives where the policy asked and reports it back — so the
simulator behaves correctly while the policy sees poses anchored where it was not trained. The symptom is a
lower eval score.

The env's own frame round-tripping (`robolab/validate.py:_check_eef_offset`, `libero/e2e.py`) is a separate
claim, and holds whether or not the constant is right.

| Integration | `env_control_frame` | Checked against `default` |
| --- | --- | --- |
| RoboLab | `models.DROID_EE_FRAME` | `robolab/validate.py:_check_flange_to_eef`, against RoboLab's own scene |
| LIBERO | identity — the grip site is taken to be `default` | `libero/validate.py:_check_flange_to_eef`, and it diverges: 38.1 mm and 90° ([#557](https://github.com/Positronic-Robotics/positronic/issues/557)) |

TODO: have the env report the offset itself, so positronic stops restating it from a robot model that is not
the env's.
