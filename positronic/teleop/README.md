# Teleoperation Guide

A person controls the robot through the teleop service, `positronic teleop`. For the Harness the person is an
ordinary `RemotePolicy`: `cfg.policy.teleop` dials the socket of the service, as a model policy dials a model server.
A human trial runs through the same eval and records the same data as a model trial.

The service runs on the same machine as the run, or on another machine (see
[Run the service on another machine](#run-the-service-on-another-machine)). It serves three things:
- the WebXR page for the phone or the headset, on port 5005;
- a window on the laptop screen, on `http://127.0.0.1:5006/`;
- one session at a time, as a model server on a Unix socket, or on a host and a port.

The service prepares each session until the person presses Ready. Each session declares stock components in its
client stack:
- `end_on_field` ends the run when a reply carries the Complete or the Give up of the person;
- `chunked_schedule` asks for one frame at a time, and plays each reply at once;
- `observation_codec` sends the task, the pose, the grip and the camera images of the robot;
- `absolute_position_action` turns each reply into a command, and `end_command` gives the `end` of a reply to
  `end_on_field`.

The service keeps the state of the trial and the clutch. Its replies carry only the target pose and the grip, or the
end. So the input of the person stays in the service. No model server gets it.

The plain form of the policy works too:

```bash
--policy=@positronic.cfg.policy.remote --policy.wire=websocket_unix \
  --policy.address=@positronic.cfg.policy.socket_address --policy.address.uds=<the socket that the service names>
```

`cfg.policy.teleop` adds a short infer timeout (10 s).

## Devices

| Device | Browser | Service config | Page |
|---|---|---|---|
| iPhone | XR Browser | `--webxr=@positronic.cfg.webxr.iphone_teleop` (http) | `http://<laptop>:5005/?mode=teleop` |
| Android phone | Chrome | `--webxr=@positronic.cfg.webxr.android_teleop` (https) | `https://<laptop>:5005/?mode=teleop` |
| Meta Quest | Oculus Browser | `--webxr=@positronic.cfg.webxr.oculus` (https, the default) | `https://<laptop>:5005/` |

The banner of `positronic teleop` prints the URL of the page, and the IPv4 address of each network interface with its
name. Use the address of the network that the phone or the headset is on. Open port 5005 in the firewall of the
laptop. With https, the browser warns about the self-signed certificate. [Data Collection](../../docs/data-collection.md) shows
how to accept the warning on the Quest.

The robot moves `sensitivity` times the distance of the controller or the phone, and turns as it turns. The phone
configs set 2, and `oculus` sets 1. `--webxr.sensitivity=<x>` changes it.

## Controls

The service asks for Ready before each human trial. Ready comes only from the phone or the headset, because it
confirms that the person holds the controller. The window on the laptop offers only Cancel.

| Phase | Phone (teleop mode) | Quest (right controller) | What it does |
|---|---|---|---|
| The trial waits for Ready | tap **Ready** | press B | Ready: the episode starts |
| The trial waits for Ready | tap **Cancel** (or Cancel in the window) | press the stick | Cancel (see Limits) |
| The trial runs | tap **Control** | press A | control on or off (the clutch) |
| Control is on | the slider | the trigger | the grip |
| The trial runs | hold **Complete** for 1 s | hold B for 1 s | Complete: the episode ends |
| The trial runs | hold **Hold: Give up** for 1 s | hold the stick for 1 s | Give up: the episode ends |

When control comes on, the hand moves the robot from its current pose, so the robot does not jump. When the phone or
the controller is not tracked for 0.5 s, control goes off.

The phone page shows the phase above its buttons. It colours Control green while control is on and grey while it is
off. The B button is labelled with what it does: Ready, or Complete while the trial runs. It pulses orange while the
trial waits for Ready, and it is green while the trial runs. At each new trial and at each end the page opens the grip
slider, so a closed gripper does not close again at the next Control.

The Quest shows the phase, the task and the control state on its video. The window on the laptop shows the same state,
and the control state.

## Cameras

A run records the union of two camera sets:
- the training set, which a model trains on;
- the operator set, which the person sees.

Conversion for training picks its own keys from the recording.

- The eval names the recorded cameras in its `camera_dict`.
- The session parameter `cameras` names the cameras that the person sees, in that order. By default the person sees
  no camera (see Session parameters).
- The window shows the cameras of the policy in a grid, each with its name.
- The headset video shows one camera: the first one, or the one that the session parameter `headset_camera` names.
  The phone page shows no video.
- Each frame request carries the camera images at 640 x 480. A camera with another aspect gets black bars.
- `positronic teleop` sends the headset video at JPEG quality 90. `--webxr.jpeg_quality=<0..100>` changes it. Lower
  it if the video stutters on a slow Wi-Fi.

## Collect demonstrations in MuJoCo

1. In terminal 1, start the service. For an iPhone:

   ```bash
   uv run positronic teleop --webxr=@positronic.cfg.webxr.iphone_teleop
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
     `image.handcam_right`. `teleop_collect` shows `image.exterior`, `image.wrist_upright`, `image.handcam_right` and
     `image.wrist`, in that order.
   - Seed `N` runs the scenes `N` to `N + 19`. So that sessions do not repeat a scene, move the seed of each session
     on by the trial count (1, 21, 41, ...), or leave out `--eval.seed` to get a random scene for each trial. Use a new
     `--output_dir` for each session too.

3. On the phone, open the page, tap **ENTER VR**, and allow the camera and the motion sensors. Hold the phone upright.
4. For each trial:
   1. When the phase line says "Human trial: tap Ready", hold the phone in a working pose and tap
      **Ready**.
   2. Tap **Control** and move the phone: the arm follows. Use the slider for the grip. Tap **Control** again to pause,
      for example to change how you hold the phone.
   3. Do the task. Then hold **Complete** for 1 s if you did it, or hold **Hold: Give up** for 1 s if you
      did not.
   4. The sim resets the scene, and the next trial waits for Ready.

The run ends after its last trial. Press Ctrl-C in terminal 2 to end it earlier.

## Collect demonstrations in MolmoSpaces

`pick_mini_collect` runs the pick mini benchmark of the MolmoSpaces suite v2. It leaves out each episode that hides
the start grasp site from the right shoulder camera. It gives each episode 120 s in place of the benchmark horizon.
It drops the object image points sensor of MolmoSpaces (`--eval.with_object_image_points=False`), which renders a
segmentation mask for each camera at each step and which no recording uses. So each step runs about 35 ms faster.
Use it for demonstrations only, not for a scored eval.

```bash
uv run positronic eval run \
  --eval=@positronic.cfg.eval.sim.molmo.pick_mini_collect --eval.episodes='[0,1,2,3,4]' \
  --policy=@positronic.cfg.policy.teleop_molmo \
  --output_dir=~/datasets/molmo_pick_demos
```

- `--eval.horizon_sec` changes the horizon of each episode. Leave out `--eval.episodes` to run all the episodes that
  the config selects.
- `teleop_molmo` shows `image.exterior`, `image.exterior_2` and `image.wrist`. The eval puts each shoulder camera in
  one exterior slot at random, so the person sees both.
- `teleop_molmo` sets `operator_position` to `BACK`, as `teleop_collect` does.

Start `positronic teleop` first, as in MuJoCo.

## Session parameters

The run sets the parameters of its session in the query of its address. The service takes no session parameters, so
one service serves runs with different parameters. A parameter that the run does not set has the default of
`positronic.teleop.protocol.SessionParams`.

| Parameter | Value | Default |
|---|---|---|
| `operator_position` | a member of `positronic.teleop.clutch.OperatorPosition`, for example `BACK`, or `None` for UMI mode, where control is always on and the hand pose is the robot pose | `FRONT` |
| `cameras` | the cameras that the person sees, in that order | none |
| `headset_camera` | the camera of `cameras` that the headset shows | the first one |
| `jpeg_quality` | the JPEG quality of the camera images | 90 |

`positronic.teleop.protocol.session_query` builds the query. In `positronic.cfg.policy`, `teleop_query` is the query
of `teleop` and `teleop_network`, and `teleop_collect_query` and `teleop_molmo_query` are the queries of
`teleop_collect` and `teleop_molmo`. Both set `operator_position` to `BACK`. Set one parameter on the command line,
for example `--policy.address.query.headset_camera=image.wrist`. The episode records the values in
`inference.policy.server.effective_params`.

## Run the service on another machine

The service can run on the laptop of the person, and the run on a rig or a VM. Then the phone and the window need no
tunnel. The machine of the run must reach the laptop on port 5007, for example on a tailnet.

1. On the laptop, start the service on a host and a port, with a token:

   ```bash
   export AUTH_TOKEN=<a long random string>
   uv run positronic teleop --webxr=@positronic.cfg.webxr.iphone_teleop \
     --served_address=@positronic.teleop.command.network
   ```

2. On the machine of the run, set the same token, and dial the laptop:

   ```bash
   export AUTH_TOKEN=<the same string>
   uv run positronic eval run \
     --eval=@positronic.cfg.eval.sim.positronic.stack_cubes_collect \
     --policy=@positronic.cfg.policy.teleop_network --policy.address.host=<laptop> \
     --policy.address.query=@positronic.cfg.policy.teleop_collect_query \
     --output_dir=~/datasets/stack_cubes_demos
   ```

- The service does not start on a host and a port with no token. A session with no token or a wrong one gets HTTP 403.
- The sessions have no TLS. Use a private network, such as a tailnet.
- Each frame waits for one round trip of the network.
- `--served_address.port` changes the port. Give `--policy.address.port` the same value.
- On the Unix socket, the service takes no token, and `AUTH_TOKEN` has no effect.

## What a trial records

| Static | Value |
|---|---|
| `inference.policy.type` | `remote` |
| `inference.policy.server.type` | `teleop` |
| `inference.policy.server.effective_params.*` | the session parameters, for example `operator_position` |
| `inference.policy.action_fps` | 100: the most frequent polls of the schedule, not the rate of the commands of the person |
| `eval.ended_by` | `policy`, when Complete or Give up ended the trial |
| `eval.policy_end` | `complete` or `give_up` |
| `eval.terminated` | `False`, when the time limit ended the trial |
| `molmo.task_horizon_override` | the horizon of `--eval.horizon_sec`, in MolmoSpaces |

For training, use the episodes with `eval.policy_end = complete`.

Each frame reply has a target. While control is off, the target is the last one, or at first the pose and the grip
that the robot has. So every command signal starts at the episode start, and the robot stays still.

## View the recordings

```bash
uv run positronic-server --dataset.path=~/datasets/stack_cubes_demos
```

Record a few trial episodes first, and view them. One task needs 50 or more demonstrations. Keep the episodes short,
and vary how you come to the object.

## Limits

- A Cancel while the trial waits for Ready ends the whole `eval run`. The end of the ready limit does the same. To skip
  a trial, tap Ready, and then hold **Hold: Give up**: the episode records a Give up.
- The phone page shows no video. Watch the window on the laptop.
- MolmoSpaces runs slowly and with jerks on a Mac, and the person feels it.
- A sim run must charge inference time, as `eval run` does by default. With `--charge_inference_time=False` the sim
  stops while each call runs, so a person cannot control the run. The service does not refuse such a run.
- The service waits for Ready while it prepares the session, before the episode starts. So the wait is outside the
  recording and the time limit. The wait blocks the control loop of the World. The service ends the wait after 120 s.
  A Ctrl-C during the wait closes the session, and the service stops the wait.
- The service serves one session at a time, and refuses a second one. Each trial opens a new session.
- When the connection drops before the first frame reply, the client opens the session again. The trial goes on with
  no second Ready. The service keeps such a trial for 45 s.
- Each frame request carries the camera images, so the images go at the rate of the frames, about the rate of the
  headset. A slow network can feel it.
- When the service is down, a trial fails after the connect deadline of the client (900 s).
- With no `XDG_RUNTIME_DIR`, the socket is in `positronic-<uid>` under the temporary directory. The service makes that
  directory private to the user.

## Check on a rig

The first frame replies ask the arm for the pose that it has. Do this check once on each rig, before the first human
trial on it. No test can run it.

1. Start `positronic teleop` and a run with `--policy=@positronic.cfg.policy.teleop` on the rig.
2. Press Ready, and do not turn control on.
3. Watch the arm and the gripper for 10 s. Neither may move.
4. Look at the `robot_command` and `target_grip` signals of the episode. Each starts at the episode start, with the
   pose and the grip that the robot had.
5. Turn control on, and move your hand slowly. The arm must follow from where it is, with no jump.

If the arm moves in step 3, stop the run, and report it before any human trial on that rig.

## See Also

- [Data Collection](../../docs/data-collection.md) – headset setup, and collection with `positronic-data-collection`
- [Inference Guide](../../docs/inference.md) – the eval run and the station console
