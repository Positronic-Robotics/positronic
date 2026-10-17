# Teleoperation Guide

A person controls the robot through the teleop service, `positronic teleop`. For the Harness the person is an
ordinary `RemotePolicy`: `cfg.policy.teleop` dials the socket of the service, as a model policy dials a model server.
A human trial runs through the same eval and records the same data as a model trial.

The service runs on the same machine as the run. It serves three things:
- the WebXR page for the phone or the headset, on port 5005;
- a window on the laptop screen, on `http://127.0.0.1:5006/`;
- one session at a time, as a model server on a Unix socket.

Each session declares `TeleopClient` (`positronic.policy.teleop`) in its client stack. The service keeps the state of
the trial and the clutch. Its replies carry only the target pose, the grip, Start task and the end. So the input of the
person stays in the service. No model server gets it.

The plain form of the policy works too:

```bash
--policy=@positronic.cfg.policy.remote --policy.wire=websocket_unix \
  --policy.address=@positronic.cfg.policy.socket_address --policy.address.uds=<the socket that the service names>
```

`cfg.policy.teleop` adds a short infer timeout (10 s) and a short connect deadline (30 s).

## Devices

| Device | Browser | Service config | Page |
|---|---|---|---|
| iPhone | XR Browser | `--webxr=@positronic.cfg.webxr.iphone_teleop` (http) | `http://<laptop>:5005/?mode=teleop` |
| Android phone | Chrome | `--webxr=@positronic.cfg.webxr.android_teleop` (https) | `https://<laptop>:5005/?mode=teleop` |
| Meta Quest | Oculus Browser | `--webxr=@positronic.cfg.webxr.oculus` (https, the default) | `https://<laptop>:5005/` |

The banner of `positronic teleop` prints the URL of the page, and the IPv4 address of each network interface with its
name. Use the address of the network that the phone or the headset is on. Open port 5005 in the firewall of the
laptop. With https, the browser warns about the self-signed certificate. [Data Collection](data-collection.md) shows
how to go past the warning on the Quest.

## Controls

The service asks for Ready before each human trial. Ready comes only from the phone or the headset, because it
confirms that the person holds the controller. The window on the laptop offers only Cancel.

| Phase | Phone (teleop mode) | Quest (right controller) | What it does |
|---|---|---|---|
| The trial waits for Ready | tap **Ready / Start task** | press B | Ready: the episode starts |
| The trial waits for Ready | tap **Cancel** (or Cancel in the window) | press the stick | Cancel (see Limits) |
| The trial runs | tap **Control** | press A | control on or off (the clutch) |
| Control is on | the slider | the trigger | the grip |
| The trial runs, before the task | tap **Ready / Start task** | press B | Start task: the real attempt starts here |
| After Start task | hold **Ready / Start task** for 1 s | hold B for 1 s | Complete: the episode ends |
| The trial runs | hold **Hold: Give up** for 1 s | hold the stick for 1 s | Give up: the episode ends |

When control comes on, the hand moves the robot from its current pose, so the robot does not jump. When the phone or
the controller is not tracked for 0.5 s, control goes off.

The phone page shows the phase above its buttons. It colours Control green while control is on and grey while it is
off. At each new trial and at each end it opens the grip slider, so a closed gripper does not close again at the next
Control. The Quest shows the phase and the task on its video.

## Cameras

A run records the union of two camera sets:
- the training set, which a model trains on;
- the operator set, which the person sees.

Conversion for training picks its own keys from the recording.

- The eval names the recorded cameras in its `camera_dict`.
- The session parameter `cameras` names the cameras that the person sees, in that order. With no `cameras`, the person
  sees every image of the observation (see Session parameters).
- The window shows the cameras of the policy in a grid, each with its name.
- The headset video shows one camera: the first one, or the one that `positronic teleop --headset_camera=<name>`
  names. The phone page shows no video.

## Collect demonstrations in MuJoCo

1. In terminal 1, start the service. For an iPhone:

   ```bash
   uv run positronic teleop --webxr=@positronic.cfg.webxr.iphone_teleop --operator_position=.BACK
   ```

2. In terminal 2, start a run of 20 trials:

   ```bash
   uv run positronic eval run \
     --eval=@positronic.cfg.eval.sim.positronic.stack_cubes_collect --eval.seed=1 \
     --policy=@positronic.cfg.policy.teleop_collect \
     --output_dir=~/datasets/stack_cubes_demos
   ```

   - `stack_cubes_collect` is `stack_cubes` for a person: 20 trials, with 120 s for each trial. The time before
     Ready does not count. Change them with `--eval.trial_count` and `--eval.timeout`.
   - It records four cameras: `image.exterior` and `image.wrist` for training, and `image.wrist_upright` and
     `image.handcam_right`. `teleop_collect` shows all four in that order.
   - Seed `N` runs the scenes `N` to `N + 19`. So that sessions do not repeat a scene, move the seed of each session
     on by the trial count (1, 21, 41, ...), or leave out `--eval.seed` to draw each scene at random. Use a new
     `--output_dir` for each session too.

3. On the phone, open the page, tap **Enter AR**, and allow the camera and the motion sensors. Hold the phone upright.
4. For each trial:
   1. When the phase line says "Human trial: tap Ready", hold the phone in a working pose and tap
      **Ready / Start task**.
   2. Tap **Control** and move the phone: the arm follows. Use the slider for the grip. Tap **Control** again to pause
      and hold the phone in a better way.
   3. Try a few moves if you like. Tap **Ready / Start task** when the real attempt starts. A reader can cut off the
      moves before it.
   4. Do the task. Then hold **Ready / Start task** for 1 s if you did it, or hold **Hold: Give up** for 1 s if you
      did not.
   5. The sim resets the scene, and the next trial waits for Ready.

The run ends after its last trial. Press Ctrl-C in terminal 2 to end it earlier.

## Collect demonstrations in MolmoSpaces

`pick_mini_collect` runs the pick mini benchmark of the MolmoSpaces suite v2. It leaves out each episode that hides
the start grasp site from the right shoulder camera. It gives each episode 120 s in place of the benchmark horizon.
Use it for demonstrations only, not for a scored eval.

```bash
uv run positronic eval run \
  --eval=@positronic.cfg.eval.sim.molmo.pick_mini_collect --eval.episodes='[0,1,2,3,4]' \
  --policy=@positronic.cfg.policy.teleop_molmo \
  --output_dir=~/datasets/molmo_pick_demos
```

- `--eval.horizon_sec` changes the horizon of each episode. Leave out `--eval.episodes` to run all the episodes that
  the config selects.
- `teleop_molmo` shows every camera of the observation. It lets the sim clock stand still for 10 s before it stops the
  run, because MolmoSpaces is slow on a laptop.
- Start the service with `--operator_position=.BACK` if the arm moves the wrong way.

Start `positronic teleop` first, as in MuJoCo.

## Session parameters

The service CLI sets the defaults: `--operator_position` (`.FRONT`, `.BACK`, or `None` for UMI mode, where control is
always on and the hand pose is the robot pose), `--cameras` (a JSON list, or none for every image), `--image_fps` and
`--still_clock_timeout`. A run sets its own in the query of its address, each value a JSON literal, for example
`--policy.address.query='operator_position="BACK"'`. `positronic.teleop.protocol.session_query` builds such a query:
`teleop_collect` and `teleop_molmo` use it. The episode records the values in `inference.policy.server.effective_params`.

## What a trial records

| Static | Value |
|---|---|
| `inference.policy.type` | `remote` |
| `inference.policy.server.type` | `teleop` |
| `inference.policy.server.effective_params.*` | the session parameters, for example `operator_position` |
| `inference.policy.attempt_start_ns` | the episode time of Start task, in ns. It is absent when the trial had no Start task. |
| `eval.ended_by` | `policy`, when Complete or Give up ended the trial |
| `eval.policy_end` | `complete` or `give_up` |
| `eval.terminated` | `False`, when the time limit ended the trial |
| `molmo.task_horizon_override` | the horizon of `--eval.horizon_sec`, in MolmoSpaces |

For training, use the episodes with `eval.policy_end = complete`.

At its first step the policy sends a stay command: the pose and the grip that the robot has. So every command signal
starts at the episode start.

## View the recordings

```bash
uv run positronic-server --dataset.path=~/datasets/stack_cubes_demos
```

Record a few trial episodes first, and view them. One task needs 50 or more demonstrations. Keep the episodes short,
and vary how you come to the object.

## Limits

- A Cancel while the trial waits for Ready ends the whole `eval run`. To skip a trial, tap Ready, and then hold
  **Hold: Give up**: the episode records a Give up.
- The phone page shows no video. Watch the window on the laptop.
- MolmoSpaces runs slowly and with jerks on a Mac. `teleop_molmo` allows for this, but the person feels it.
- The policy waits for Ready before the episode starts, so the wait is outside the recording and the time limit. The
  wait blocks the main group of the World. The client asks again each second. The service ends the wait after 120 s,
  and the client stops after 130 s. A Ctrl-C during the wait takes effect after the next answer.
- The service serves one session at a time, and refuses a second one. Each trial opens a new session. The `meta()` of
  a run opens a short session too, so start no second run while a trial runs.
- When the service is down, a trial fails after the connect deadline of the policy (30 s in `cfg.policy.teleop`).
- With no `XDG_RUNTIME_DIR`, the socket is in `positronic-<uid>` under the temporary directory. The service makes that
  directory private to the user.

## Check on a rig

The stay command asks the arm for the pose that it has. Do this check once on each rig, before the first human trial
on it. No test can run it.

1. Start `positronic teleop` and a run with `--policy=@positronic.cfg.policy.teleop` on the rig.
2. Press Ready, and do not turn control on.
3. Watch the arm and the gripper for 10 s. Neither may move.
4. Look at the `robot_command` and `target_grip` signals of the episode. Each starts at the episode start, with the
   pose and the grip that the robot had.
5. Turn control on, and move your hand slowly. The arm must follow from where it is, with no jump.

If the arm moves in step 3, stop the run, and report it before any human trial on that rig.

## See Also

- [Data Collection](data-collection.md) – headset setup, and collection with `positronic-data-collection`
- [Inference Guide](inference.md) – the eval run and the station console
