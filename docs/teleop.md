# Teleoperation Guide

`TeleopPolicy` lets a person control the robot through a VR headset. For the Harness it is an ordinary policy: a
human trial records the same data as a model trial, and the same eval drives it.

The person works through the teleop service, `positronic teleop`. The service runs beside the run, on the same
machine. It serves the WebXR page of the headset, a window on the station screen, and one session for the policy on a
Unix socket. The headset input reaches only the policy. No model server gets it.

## Teleoperate in sim

Start the service in one terminal:

```bash
uv run positronic teleop
```

It prints the URL of the headset page (port 5005, HTTPS) and opens the window on the station screen
(`http://127.0.0.1:5006/`).

Run an eval with the `teleop` policy in a second terminal:

```bash
uv run positronic eval run \
  --eval=@positronic.cfg.eval.sim.positronic.stack_cubes --eval.timeout=120 \
  --policy=@positronic.cfg.policy.teleop \
  --output_dir=~/datasets/teleop_demo
```

- `--eval.timeout` is the time limit of each trial, in seconds. The default of `stack_cubes` is 15 s, which is short
  for a person.
- If you stand behind the robot, add `--policy.operator_position=@positronic.teleop.clutch.OperatorPosition.BACK`.
- Keep the charged inference time of `eval run` on (its default). A person needs the sim clock to run with the wall
  clock. With an uncharged clock the run fails after the clock stands still for one second.

In the Quest, open the URL in the Oculus Browser and enter AR. The browser warns about the self-signed certificate.
[Data Collection](data-collection.md#vr-teleoperation-meta-quest) shows the steps.

## Controls

The service asks for Ready before each human trial. Then the trial runs until you end it, the env ends it, or the time
limit passes.

| Control | When | What it does |
|---|---|---|
| B, or Ready in the window | the trial waits for Ready | Ready |
| Stick, or Cancel in the window | the trial waits for Ready | Cancel. The trial does not start (see Limits). |
| A | the trial runs | control on or off (the clutch) |
| Trigger | control is on | the grip |
| B | before the task starts | Start task: the real attempt starts here |
| Hold B for 1 s | after Start task | Complete |
| Hold the stick for 1 s | the trial runs | Give up |

When control comes on, the hand moves the robot from its current pose, so the robot does not jump. When the headset
loses the controller for 0.5 s, control goes off.

The headset video and the window show the camera image of the run (`image.exterior` by default), with the task and
the state written on it. The window also shows "Human trial" when a trial waits for Ready.

`positronic teleop --help` lists the settings: `--ready_limit`, `--silence_limit`, `--hold_sec`, `--press_threshold`,
`--view_fps`, `--window_port`, `--open_window` and `--webxr`.

## What a human trial records

A human trial records the signals and the statics of every trial. These statics are particular to it:

| Static | Value |
|---|---|
| `inference.policy.type` | `teleop` |
| `inference.policy.attempt_start_ns` | the episode time of Start task, in ns. It is absent when the trial had no Start task. |
| `eval.ended_by` | `policy`, when Complete or Give up ended the trial |
| `eval.policy_end` | `complete` or `give_up` |
| `eval.terminated` | `False`, when the time limit ended the trial |

The moves before `attempt_start_ns` are trial moves. A reader can cut them off. The Harness records them as it records
the rest of the episode.

At its first step the policy sends a stay command: the pose and the grip that the robot has. So every command signal
starts at the episode start.

## Limits

- The policy waits for Ready before the episode starts, so the wait is outside the recording and the time limit. The
  wait blocks the main group of the World. The service answers an error after the ready limit (120 s), and the policy
  stops waiting after its ready timeout (130 s). A Ctrl-C during the wait takes effect after the answer.
- A Cancel or a missed ready limit fails the episode before its first step, and nothing is recorded. `eval run` runs a
  trial again only after a device gives no data, so a Cancel ends the run. Start the run again to go on.
- The service serves one session at a time.
- On a station with no `XDG_RUNTIME_DIR`, the socket is in `positronic-<uid>` under the temporary directory. The service
  makes that directory private to the user.

## Check on a rig

The stay command asks the arm for the pose that it has. On a Franka, the driver solves the joints for that pose. Do
this check once on each rig before the first human trial on it. No test can run it.

1. Start `positronic teleop` and a run with `--policy=@positronic.cfg.policy.teleop` on the rig.
2. Press Ready, and do not press A.
3. Watch the arm and the gripper for 10 s. Neither may move.
4. Look at the `robot_command` and `target_grip` signals of the episode. Each starts at the episode start, with the
   pose and the grip that the robot had.
5. Press A, and move your hand slowly. The arm must follow from where it is, with no jump.

If the arm moves in step 3, stop the run and report it before any human trial on that rig.

## See Also

- [Data Collection](data-collection.md) – headset setup, and collection with `positronic-data-collection`
- [Inference Guide](inference.md) – the eval run and the station console
