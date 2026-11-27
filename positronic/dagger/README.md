# DAgger rounds on a rig

The round driver, `run` in [`rounds.py`](rounds.py), runs DAgger (dataset aggregation) rounds. Its config gives a rig
with its teleoperator, a series of tasks and a model. In simulation, `mujoco_lift_cube` (on CPU) and
`robolab_banana_in_bowl` (on a GPU host) in [`cfg/dagger.py`](../cfg/dagger.py) give them. This document names each part
that a real rig gives in place of the simulation parts, and where each part plugs in.

## What stays the same

On a real rig, these parts do not change:

- `run` and its four steps: Collect, Extract, Train and Run again. Its folders, and its checks of the tasks and of the
  licenses, stay.
- `RoundZeroPolicy`, the policy of round 0. At each step it commands the joints and the grip of each arm that the rig
  measured at the first step.
- The harness ([`policy/harness.py`](../policy/harness.py)) with the takeover of the helper
  ([`policy/takeover.py`](../policy/takeover.py)), and the record with the step number `harness.step`.
- The DAgger reader, `DaggerDataset` ([`policy/dagger.py`](../policy/dagger.py)). It finds the samples of a step by
  `harness.step`. So a rig with a wall clock gives training episodes in the same form as a simulator.
- `write_training_data` ([`training_data.py`](training_data.py)), `ModelCommands`, and the LeRobot ACT commands of
  `lerobot_act` ([`vendors/lerobot_0_3_3`](../vendors/lerobot_0_3_3/README.md)).
- The remote policy, `RemotePolicy`, which drives the Collect step of each round after round 0.

## The parts of a rig

| Part | Where it plugs in | In positronic |
|---|---|---|
| [The arm drivers](#the-arm-drivers) | `Embodiment` ([`eval/__init__.py`](../eval/__init__.py)) | Yes |
| [The gripper drivers](#the-gripper-drivers) | `Embodiment` | Yes |
| [The camera drivers](#the-camera-drivers) | `Embodiment` | Yes |
| [The teleop device](#the-teleop-device) | `Teleoperator` ([`policy/teleoperator.py`](../policy/teleoperator.py)), in `Rig.harness` | No |
| [The rig operator](#the-rig-operator) | The `scene` prepare handler and the `done` signal of the harness | No |
| [The tasks](#the-tasks) | `tasks` of `run`, as `Task` ([`eval/__init__.py`](../eval/__init__.py)) | No |
| [The host of the trainer and of the policy server](#the-host-of-the-trainer-and-of-the-policy-server) | `model` of `run`, as `ModelCommands` ([`rounds.py`](rounds.py)) | Yes, on the host of the round driver |
| [The `Rig`](#the-rig) | `rig` of `run`, as `Rig` ([`rounds.py`](rounds.py)) | Only for the simulators |
| [The identities and the licenses](#the-identities-and-the-licenses) | `meta()` of each source, and `inference.<name>` in the record | In part |

### The arm drivers

- **Where it plugs in.** The harness drives the rig through an `Embodiment`. An arm driver is a `pimm.ControlSystem` in
  `Embodiment.control_systems`. It takes the commands on `robot_command.<arm>`, by the names of `keys.arm_channel`. A
  rig with one arm uses the bare names, such as `robot_command`.
- **The `Embodiment`.** The rig puts the signals of each driver in `Embodiment.observations` and
  `Embodiment.commands`, and the frames and the pose signals in `Embodiment.static_meta`.
  [`cfg/embodiment.py`](../cfg/embodiment.py) makes the `Embodiment` of each rig of positronic, for example
  `yam_bimanual`. `remote_embodiment` in [`simulator/env_server/proxy.py`](../simulator/env_server/proxy.py) makes the
  `Embodiment` of a simulated rig in the same form.
- **What it gives.** `RoundZeroPolicy` needs the joints, `robot_state.<arm>.q`. The codec `ee` of `lerobot_act`
  needs the pose, `robot_state.ee_pose`. `ee` is for one arm. The codec `ee_bimanual` is for the two-arm YAM. It needs
  the pose and the grip of each arm, `robot_state.<arm>.ee_pose` and `grip.<arm>`.
- **What it takes.** `RoundZeroPolicy` sends a `JointPosition`. `lerobot_act` learns the label `robot_command.pose`, or
  `robot_command.<arm>.pose` of each arm with `ee_bimanual`.
  So the teleop device sends a `CartesianPosition`, and ACT sends one too. Each arm takes both kinds of command.
- **Pose commands.** The harness refuses a pose command of a source that states no frame for that pose. The rig states
  its frames in `Embodiment.static_meta`, with `ee_frames_entry`
  ([`drivers/roboarm/models.py`](../drivers/roboarm/models.py)). `lerobot_act` states its frame at the site `gripper`.
  So the rig states its frames at `gripper`.
- **Errors.** An arm that can hold an error gives a handler in `Embodiment.ready_handlers`. The harness calls it
  before each episode.
- **In positronic.** The drivers [`roboarm/yam`](../drivers/roboarm/yam), [`roboarm/franka.py`](../drivers/roboarm/franka.py),
  [`roboarm/kinova`](../drivers/roboarm/kinova) and [`roboarm/so101`](../drivers/roboarm/so101).

### The gripper drivers

- **Where it plugs in.** A gripper driver gives the observation `grip.<arm>` and takes the command `target_grip.<arm>`.
  `RoundZeroPolicy` reads `grip.<arm>` at the first step.
- **In positronic.** The drivers in [`gripper`](../drivers/gripper), for example `dh.py` and `robotiq.py`.

### The camera drivers

- **Where it plugs in.** A camera driver gives the observation `image.<name>`. The codec `ee` of `lerobot_act` reads
  `image.wrist` and `image.exterior`. `ee_bimanual` reads `image.exterior`, `image.wrist_left` and `image.wrist_right`.
  So a rig gives its cameras these names, or the config gives a codec with the names of the rig.
- **In positronic.** The drivers in [`camera`](../drivers/camera), for example `zed.py`, `luxonis.py` and
  `linux_video.py`.

### The teleop device

In simulation, the scripted expert is the helper. On a rig, a person is the teleoperator. The helper gets no ground
truth on a rig, because a rig has no privileged signals.

- **Where it plugs in.** `Teleoperator(inputs, license)` reads the console on a `pimm.SignalReceiver`. Each message
  holds the commands and the press fields `take_over` and `hand_back`. A press field holds the time of the last press,
  so a new value is a new press. `license` is the license of the commands of the person
  ([The identities and the licenses](#the-identities-and-the-licenses)).
- **The presses.** Take over gives the rig to the teleoperator in the step of the press, and Hand back gives it back to
  the policy. The first step of a run gives the last message of the console, so a press from the episode before is not
  new. While the teleoperator drives, a step with no new message gives the rig the commands of the newest message.
- **Pose commands.** The harness reads each pose command of the helper in the frame that the rig states. So the device
  sends each pose at the site `gripper`, not at the flange or the wrist.
- **The start of control.** At Take over, the rig gets the commands of the message of the press. So the device sends
  commands at the current pose of the rig before the press. `Teleoperator` does not give the observation to the device.
  So the device reads the arm state of the rig itself.
- **Round 0.** `run` calls `Rig.harness` with `takes_over_at_start=True` in round 0, and with `False` after. On a rig,
  the person presses Take over at the start of each episode of round 0. A press in the first message is not new, so the
  person presses after the episode starts. So the `Rig` of a real rig takes this argument and does not use it.
- **Wiring.** Each Collect step opens a new `pimm.World`. `Rig.harness` makes a new teleop device in it, and connects
  the arm state to the device. `Teleoperator` is not a control system, so a control system of the rig owns the receiver
  that it reads. `Rig.harness` connects the device to that receiver, and gives a new `Teleoperator` of the receiver to
  `run`. It starts the device and the owner of the receiver with the harness and the drivers. A pimm receiver connects
  to one emitter only, so a device of one world does not connect in the next.
- **In positronic.** [`drivers/webxr.py`](../drivers/webxr.py) gives the poses and the buttons of a headset or a phone.
  The teleop service in [`teleop`](../teleop/README.md) keeps the clutch itself. It serves the person as a remote policy
  with only a pose, a grip and an end. It gives no press fields, so it does not plug into `Teleoperator` with no change.

### The rig operator

The rig operator is a person at the rig. In simulation, the simulator does the work of this person.

- **The scene reset.** The rig operator resets the scene between episodes. The harness asks each handler that
  `Task.prepare_args` names before an episode, and starts the episode when each one answers. So the `scene` handler of
  the rig answers when the rig operator says that the scene is ready.
- **The start pose.** An `arm` prepare handler moves the arm to the start pose of an episode.
- **The end of an episode.** The rig operator can end an episode through the `done` signal of `wire_embodiment`
  ([`wire.py`](../wire.py)). A truthy `done` ends the episode, and its payload goes into the answer. So a payload with
  `eval_keys.SUCCESS` gives the success of the episode.
- **In positronic.** [`drivers/keyboard.py`](../drivers/keyboard.py) reads a key from the terminal. No `scene` handler
  and no `done` input for a person exist.

### The tasks

- **Where it plugs in.** `run` takes a series of `Task`. Each episode takes the next task. A run stops with an error
  before the Collect step of a round that gets too few tasks.
- **A rig.** A task of a rig gives the instruction, the time limit, and the prepare arguments of the `scene` and of each
  arm.
- **The end of an episode.** An episode ends at the time limit of its task, at the `done` signal of the rig operator,
  or when the run of the policy ends. `RoundZeroPolicy` does not end its run. So each task of a rig sets
  `timeout_sec`, or the rig gives a `done` signal.
- **An error.** An error in an episode, such as a failed prepare handler, stops `run`. The round gets no Extract step.
  `run` refuses an output folder that has files, so a new run starts at round 0 in a new folder.
- **Success.** `run` counts the episodes whose answer has `eval_keys.SUCCESS`. A rig with no `done` signal counts none.
  This does not change the training data.

### The host of the trainer and of the policy server

- **Where it plugs in.** `ModelCommands` gives the commands of the converter, the trainer and the policy server as lists
  of arguments. `run` runs them as subprocesses on its own host.
- **The policy server.** `run` binds the server on `localhost`. So the policy server runs on the host of the round
  driver, and the harness of the rig reaches it there.
- **The trainer.** The trainer also runs on that host, in the environment of the round driver with the
  `lerobot_0_3_3` extra ([LeRobot ACT](../vendors/lerobot_0_3_3/README.md)). That README names what the host needs.
- **Another host.** A trainer and a server on another host need a change of `run`.

### The `Rig`

`Rig` in [`rounds.py`](rounds.py) is the rig of the Collect step. `MujocoRig` in [`mujoco.py`](mujoco.py) runs
`MujocoSim` in the process, and `EnvServerRig` in [`env_server.py`](env_server.py) runs an env server. A real rig
implements these three members:

- **`observations`.** The observation keys of the `Embodiment` of the rig. The DAgger reader keeps them as the
  observation of a step.
- **`commands`.** The command channels of the `Embodiment` of the rig. `RoundZeroPolicy` holds each arm channel at its
  joints, and each grip channel at its `grip.<arm>` observation.
- **`harness(*, takes_over_at_start)`.** A context manager. It opens a `pimm.World` with the system clock, not
  `virtual_time`. It makes the `Embodiment` from the drivers, with `simulated=False`, and a `Harness`. It calls
  `wire_embodiment` with no `privileged` signals, and with the `done` signal of the rig operator. It makes the teleop
  device and its `Teleoperator` in the world ([The teleop device](#the-teleop-device)). `run_harness` in
  [`rounds.py`](rounds.py) runs the world. The value of `harness` is a pair: a function that runs one `Rollout` and
  returns the answer of `Harness.perform_task`, and the `Teleoperator`.
- **The config.** A config of the rig goes beside `mujoco_rig` in [`cfg/dagger.py`](../cfg/dagger.py). It gives the
  cameras, the factories of the drivers and of the teleop device, and the license of the teleoperator. A config makes
  each value once, and a pimm receiver connects to one emitter only. So `Rig.harness` calls the factories in each world,
  before the world starts: the world connects the signals when it starts. A `run.override` beside `mujoco_lift_cube`
  gives the rig, the tasks and the model. [`__main__.py`](__main__.py) names it.

No `Rig` for a real rig exists.

### The identities and the licenses

Each source states its identity and its license in its `meta()`, or a remote source in its server metadata. The harness
writes the `meta()` of the policy under `inference.policy` and of the helper under `inference.teleoperator`. The
metadata of a remote server goes under `inference.<name>.server`.

| Source | Identity | License | On a rig |
|---|---|---|---|
| `policy` in round 0 | `RoundZeroPolicy.meta()`: `type` is `dagger_round_0` | `round_0_license` of the config of `run` | The same |
| `policy` after round 0 | The metadata of the ACT server: `type` is `act` | `--model.license`, from the `license` of `lerobot_act` | The same. The license depends on the training data and the backbone weights, so the owner of the records of the rig sets it |
| `teleoperator` | `Teleoperator.meta()`: `type` is `teleoperator` | The `license` of `Teleoperator`, from the config of the rig | `Teleoperator` states no identity of the person |

In simulation, the scripted expert states the license that the config of the rig gives it: `license` of `mujoco_rig`,
or `license` of `banana_in_bowl_expert`.

`run` takes `training_licenses`, the licenses that allow training. The DAgger reader drops each record in which a
source that drove or that a step used has a license that is not in that set, a missing license, or no
`inference.<name>` entry ([`policy/license.py`](../policy/license.py)). `run` stops with an error before the first
episode when `training_licenses` does not have the license of the round 0 policy, of the model or of the teleoperator.
