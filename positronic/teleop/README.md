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
- `inference_schedule` asks for one frame at a time; `trajectory_player` plays each reply at once;
- `change_ee_frame` states the grip point `gripper` as the frame of the pose and of the target of each arm;
- `observation_codec` sends the task, the pose and the grip of each arm, and the camera images of the robot;
- `absolute_position_action` turns each reply into a command, or `bimanual_cartesian_action` into one command for each
  arm of a bimanual session (see [Two arms](#two-arms)), and `end_command` gives the `end` of a reply to `end_on_field`.
- `encode_images` sends each camera image as JPEG. With the session parameter `video_bitrate`, `encode_video` sends
  each camera as an H.264 stream, and the service decodes it.

The service keeps the state of the trial and the clutches. Its replies carry only the target pose and the grip of each
arm, or the end. So the input of the person stays in the service. No model server gets it.

The plain form of the policy works too:

```bash
--policy=@positronic.cfg.policy.remote --policy.wire=websocket_unix \
  --policy.address=@positronic.cfg.policy.socket_address --policy.address.uds=<the socket that the service names>
```

`cfg.policy.teleop` adds a short infer timeout (10 s).

## Devices

| Device | Browser | Service config | Page |
|---|---|---|---|
| iPhone | XR Browser | `--webxr=@positronic.cfg.webxr.iphone` (http) | `http://<laptop>:5005/` |
| Android phone | Chrome | `--webxr=@positronic.cfg.webxr.android` (https) | `https://<laptop>:5005/` |
| Meta Quest | Oculus Browser | `--webxr=@positronic.cfg.webxr.oculus` (https, the default) | `https://<laptop>:5005/` |
| PlayStation VR2 | Chrome on the PC | `--webxr=@positronic.cfg.webxr.psvr2` (https) | `https://localhost:5005/` |

The PlayStation VR2 has no browser of its own. It connects to a PC through the PC adapter and SteamVR, and Chrome on
that PC opens the page. The address above is for a service that runs on the same PC; on another machine, use its
address.

The banner of `positronic teleop` prints the URL of the page, and the IPv4 address of each network interface with its
name. Use the address of the network that the phone or the headset is on. Open port 5005 in the firewall of the
laptop. The headset video comes over WebRTC on UDP ports that the laptop chooses, so the firewall must also let UDP
from the headset in. With https, the browser warns about the self-signed certificate. [Data Collection](../../docs/data-collection.md) shows
how to accept the warning on the Quest.

The robot moves `sensitivity` times the distance of the controller or the phone, and turns as it turns. The phone
configs set 2, and `oculus` and `psvr2` set 1. `--webxr.sensitivity=<x>` changes it.

## Controls

The service asks for Ready before each human trial. Ready comes only from the phone or the headset, because it
confirms that the person holds the controller. The window on the laptop offers only Cancel.

| Phase | Phone (teleop mode) | Quest (right controller) | PlayStation VR2 (right controller) | What it does |
|---|---|---|---|---|
| The trial waits for Ready | tap **Ready** | press B | press Circle (○) | Ready: the episode starts |
| The trial waits for Ready | tap **Cancel** (or Cancel in the window) | press the stick | press the stick | Cancel (see Limits) |
| The trial runs | tap **Control** | press A | press Cross (✕) | control on or off (the clutch) |
| Control is on | the slider | the trigger | the trigger | the grip |
| The trial runs | hold **Complete** for 1 s | hold B for 1 s | hold Circle (○) for 1 s | Complete: the episode ends |
| The trial runs | hold **Hold: Give up** for 1 s | hold the stick for 1 s | hold the stick for 1 s | Give up: the episode ends |

When control comes on, the hand moves the robot from its current pose, so the robot does not jump. A turn of the hand
turns the gripper about its grip point. When the phone or the controller is not tracked for 0.5 s, control goes off.

### Two arms

A session with `arms=BIMANUAL` drives two arms, for example the bimanual YAM. The left controller drives the left arm,
and the right controller drives the right arm. Each arm has its own clutch. The other controls stay on the right
controller, as for one arm.

| Phase | Quest (left controller) | Quest (right controller) | What it does |
|---|---|---|---|
| The trial waits for Ready | | press B | Ready: the episode starts |
| The trial waits for Ready | | press the stick | Cancel (see Limits) |
| The trial runs | press X | | control of the left arm on or off |
| The trial runs | | press A | control of the right arm on or off |
| Control of the arm is on | the trigger | the trigger | the grip of the arm |
| The trial runs | | hold B for 1 s | Complete: the episode ends |
| The trial runs | | hold the stick for 1 s | Give up: the episode ends |
| Each phase | press or hold Y | | the camera and the panel (see Cameras) |

When the left controller is not tracked for 0.5 s, control of the left arm goes off. The right arm does the same with
the right controller. The headset video shows the control state of each arm on its own line, for example
`Left on (X: off)` and `Right off (A: on)`. When no controller data comes in the silence limit, the line gives the
cause, for example `Control off: connection lost (A: on)`. Control stays off when the data comes again, until a press of
A or X. The service logs each change of the phase and of control, with its cause. The window and the phone page show
"Control on" while control of at least one arm is on.

The phone is one controller, so a phone drives only a one-arm session. In a one-arm session, the left controller
drives no arm.

The phone page shows the phase above its buttons. It colours Control green while control is on and grey while it is
off. The B button is labelled with what it does: Ready, or Complete while the trial runs. It pulses orange while the
trial waits for Ready, and it is green while the trial runs. At each new trial and at each end the page opens the grip
slider, so a closed gripper does not close again at the next Control.

The Quest shows the phase, the task and the control state on its video. While a trial waits for Ready, the video shows
no task. From Ready to the first frame request of the run, the video shows "Trial starting: waiting for the robot" and
no control state. The control state of an arm is on green while control is on, and on yellow while it is off. The window
on the laptop shows the same state, and the control state.

The page sends controller data only in VR, so outside VR a press of B does nothing. `tracked` in `GET /status` of the
window and in the page state is false when no controller data came in the silence limit (0.5 s). While a trial waits
for Ready or runs, the window and the headset page show a notice when `tracked` is false. To send controller data again:
1. On the headset page, press ENTER VR.
2. If the button shows EXIT VR, press it, reload the page, and press ENTER VR again.

While a trial waits for Ready, the service shows a dialog with a sound on the screen of its computer. The dialog
comes also when the window is not open, and the service closes it when the wait ends. On macOS the service uses
`osascript` and `afplay`, and the dialog is in the centre of the screen. On Linux it uses `zenity` on `DISPLAY`, and
plays the sound with `paplay`. The window manager puts the dialog, for example in the top left corner.
`--notice.sound=<file>` changes the sound, and `--notice=None` turns the notice off.

A Linux service that runs as a user other than the user at the screen shows the dialog only with `DISPLAY=:0` and an X
grant from the session at the screen: `xhost +SI:localuser:<user>`. The sound of `paplay` then plays nowhere, because
only the user at the screen has the sound output.

## Cameras

A run records the union of two camera sets:
- the training set, which a model trains on;
- the operator set, which the person sees.

Conversion for training picks its own keys from the recording.

- The eval names the recorded cameras in its `camera_dict`.
- The session parameter `cameras` names the cameras that the person sees, in that order (see Session parameters).
- A session that names no cameras gets the cameras of the station. Name them once when you start the service:
  `positronic teleop --cameras='["image.exterior", "image.wrist"]'`. Each name must be a camera that the runs of the
  station send, or the run fails at its first frame. With neither, the person sees no camera. A session that names
  `headset_camera` must also name `cameras`.
- The window shows the cameras of the policy in a grid, each with its name.
- The headset video shows all the cameras of the session. The main camera fills the left part. A column on the right
  shows each other camera, with its name. The teleoperator's last choice sets the main camera. With no choice, the
  session parameter `headset_camera` sets it, else the first camera. The choice holds over the trials.
- The panel in the headset has the aspect of the video: 4:3 for one camera, 2:1 with the column.
- The headset page (before Enter AR) shows the state of the cameras: each camera, a green dot on each camera whose
  images came in the last 30 s (also between trials), and the camera that the headset shows.
- In the headset, Y on the left controller works the video panel. On the PlayStation VR2 it is Triangle (△). A press
  of Y makes the next camera whose images come now the main camera. A hold of 1 s cycles the panel: on black with the
  camera, on black with the camera and no task (the control line and the hints stay), the text alone over the room
  (the service then sends the view with no camera, and the page shows black as clear), and hidden. A page that loads
  starts on black with the task, and the stage holds over the trials. The bottom of the view names the camera and the
  use of Y, with the Quest names as for B and A.
- The phone page shows no video.
- Each frame request carries the camera images at 640 x 480, as JPEG. A camera with another aspect gets black bars.
- A session that sets `video_bitrate` sends each camera as an H.264 stream in place of JPEG. H.264 sends only the
  changes between frames, so each frame request is smaller. Set `video_fps` to the frame rate of the cameras. Such a
  session asks for frames at most `video_fps` times a second, so the target of the robot updates at most that often.
- `positronic teleop` sends the headset video at up to 30 frames per second, as one WebRTC video track.
  `--webxr.video_fps=<frames per second>` changes the frame rate. The bit rate follows the estimate of the headset
  browser, so a slow Wi-Fi lowers the quality.
- Each 5 s the service logs the video that the headset page shows: the frames, the age of a frame from the encoder
  to the page, and the stalls (`Video over WebRTC: ...`).

## Collect demonstrations in MuJoCo

1. In terminal 1, start the service. For an iPhone:

   ```bash
   uv run positronic teleop --webxr=@positronic.cfg.webxr.iphone
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
  --eval=@positronic.cfg.eval.sim.molmo.pick_mini_collect \
  --eval.episodes='[367,468,472,486,508,521,650,663,703,787]' \
  --policy=@positronic.cfg.policy.teleop_molmo \
  --output_dir=~/datasets/molmo_pick_demos
```

- No eval that the competition scores has the 10 episodes of the example, or their houses. Their physics runs
  fast. `@positronic.cfg.ds.sim.molmo_pick_demos` holds one demonstration of each.
- `--eval.horizon_sec` changes the horizon of each episode. Leave out `--eval.episodes` to run all the episodes that
  the config selects.
- `teleop_molmo` shows `image.exterior`, `image.exterior_2` and `image.wrist`. The eval puts each shoulder camera in
  one exterior slot at random, so the person sees both.
- `teleop_molmo` sets `operator_position` to `BACK`, as `teleop_collect` does.

Start `positronic teleop` first, as in MuJoCo.

## Collect demonstrations on the bimanual YAM in ABC

`teleop_yam` drives the two arms of the bimanual YAM. It shows `image.exterior`, `image.wrist_left` and
`image.wrist_right`: the cameras of the ABC eval and of `cfg.embodiment.yam_bimanual`. It sets `operator_position` to
`BACK`: stand behind the arms and look the way they look, so that the left arm is on your left.

1. Start `positronic teleop` with the Quest config, as in MuJoCo.
2. Start a run of the ABC task `put_plastic_bottles_in_bin`:

   ```bash
   uv run positronic eval run \
     --eval=@positronic.cfg.eval.sim.amazon_abc.put_bottles --eval.timeout=120 \
     --policy=@positronic.cfg.policy.teleop_yam \
     --output_dir=~/datasets/abc_bottles_demos
   ```

   The first run prepares the ABC environment and its assets, which takes some minutes.
3. In the headset, press B for Ready. Press X and A to turn on control of the left arm and the right arm. Use the
   two triggers for the two grips. Hold B for 1 s for Complete.

A rig run uses the same policy, with the eval of the rig in place of `amazon_abc`.

## Time a run with no person

`positronic.teleop.scripted_headset` does the trials in place of the person and the headset. It connects to the page
of the service, and sends a sample of the right controller at 72 Hz, as a Quest does. In each trial it presses
Ready, turns control on, and moves the hand on a circle of 5 cm for 20 s. Then it holds B for Complete.

1. Start `positronic teleop`.
2. Start a run with `--timing=True`, for example the MuJoCo run of 2 trials with `--eval.trial_count=2`.
3. Start the scripted headset with the same number of trials:

   ```bash
   uv run python -m positronic.teleop.scripted_headset --page=https://127.0.0.1:5005 --trials=2
   ```

4. After the run, read the report: `positronic eval timing-report --run_dir=<the output_dir of the run>`.

`--rate_hz`, `--trial_sec`, `--radius` and `--circle_sec` change the samples and the motion. When the service runs
with `--webxr.video_fps`, give the scripted headset the same `--view_fps`. The scripted headset holds each button until
the page shows the effect of the press. Then it releases the button for at least `--min_hold_sec` (0.2 s): raise it on a
loaded machine. A press of Ready that the page does not show within 1 s is released and pressed again. Use `http` in `--page` for a phone config of the service. The scripted headset does not check the
self-signed certificate of the page.

## Session parameters

The run sets the parameters of its session in the query of its address. The service takes no session parameters, so
one service serves runs with different parameters. A parameter that the run does not set has the default of
`positronic.teleop.protocol.SessionParams`.

| Parameter | Value | Default |
|---|---|---|
| `operator_position` | a member of `positronic.teleop.clutch.OperatorPosition`, for example `BACK`, or `None` for UMI mode, where control is always on and the hand pose is the robot pose | `FRONT` |
| `arms` | a member of `positronic.teleop.protocol.Arms`: `ONE` for one arm, on the right controller, or `BIMANUAL` for two arms (see [Two arms](#two-arms)) | `ONE` |
| `cameras` | the cameras that the person sees, in that order | none |
| `headset_camera` | the camera of `cameras` that the headset shows | the first one |
| `jpeg_quality` | the JPEG quality of the camera images | 90 |
| `video_bitrate` | the bit rate of the camera images as H.264, in bits per second, or `None` for JPEG | `None` |
| `video_fps` | the frame rate that the H.264 encoder spreads `video_bitrate` over | 30 |

`positronic.teleop.protocol.session_query` builds the query. In `positronic.cfg.policy`, `teleop_query` is the query
of `teleop` and `teleop_network`, and `teleop_collect_query`, `teleop_molmo_query` and `teleop_yam_query` are the
queries of `teleop_collect`, `teleop_molmo` and `teleop_yam`. All three set `operator_position` to `BACK`.
`teleop_yam_query` also sets `arms` to `BIMANUAL`. Set one parameter on the command line,
for example `--policy.address.query.headset_camera=image.wrist`. The episode records the values in
`inference.policy.server.effective_params`.

## Run the service on another machine

The service can run on the laptop of the person, and the run on a rig or a VM. Then the phone and the window need no
tunnel. The machine of the run must reach the laptop on port 5007, for example on a tailnet.

1. On the laptop, start the service on a host and a port, with a token:

   ```bash
   export AUTH_TOKEN=<a long random string>
   uv run positronic teleop --webxr=@positronic.cfg.webxr.iphone \
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

- A session with no token or a wrong one gets HTTP 403.
- With `AUTH_TOKEN` unset, the service serves open: each peer that reaches the port can open a session. Then dial it
  with `--policy.headers=None`.
- The sessions have no TLS. Use a private network, such as a tailnet.
- A plan file sends the token with `auth_token_file` on the endpoint. See the plan-file section of
  [client/README.md](../../client/README.md). The `websocket`, `grpc` and `roboarena` wires have no TLS, so the token
  travels in clear on them. Keep the service on a private network, such as a tailnet, or put it behind a TLS wire.
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
uv run positronic-dataset-viewer --dataset.path=~/datasets/stack_cubes_demos
```

Record a few trial episodes first, and view them. One task needs 50 or more demonstrations. Keep the episodes short,
and vary how you come to the object.

## Limits

- A Cancel while the trial waits for Ready ends the whole `eval run`. The end of the ready limit does the same. To skip
  a trial, tap Ready, and then hold **Hold: Give up**: the episode records a Give up. When no controller data came in
  the wait, the error of the ready limit says so.
- The phone page shows no video. Watch the window on the laptop.
- MolmoSpaces runs slowly and with jerks on a Mac, and the person feels it.
- A sim run must charge inference time, as `eval run` does by default. With `--charge_inference_time=False` the sim
  stops while each call runs, so a person cannot control the run. The service does not refuse such a run.
- The service waits for Ready while it prepares the session, before the episode starts. So the wait is outside the
  recording and the time limit. The wait blocks the control loop of the World. The service ends the wait after 120 s.
  A Ctrl-C during the wait closes the session, and the service stops the wait.
- The service serves one session at a time, and refuses a second one. Each trial opens a new session.
- The scripted headset sends only the right controller, so it does the trials of one-arm sessions only.
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

On a bimanual rig, use `--policy=@positronic.cfg.policy.teleop_yam`, and do steps 3 to 5 for each arm. The signals are
`robot_command.left`, `target_grip.left`, `robot_command.right` and `target_grip.right`.

## See Also

- [Data Collection](../../docs/data-collection.md) – headset setup, hardware setup and review of the data
- [Inference Guide](../../docs/inference.md) – the eval run and the station console
