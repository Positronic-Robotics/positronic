# ASPIRE CapApi programs

[ASPIRE](https://github.com/NVlabs/ASPIRE) agents write Python programs against its CapApi: `move_to_pose`,
`set_gripper`, `get_robot_state` and other calls. This vendor runs such a program as an ordinary remote policy. The rig
sends observations and plays the action chunks that the server answers. The program never drives a motor. The rig's
driver solves IK and limits speed.

The vendor holds a port of the ASPIRE value types (`contracts.py`) and does not depend on the ASPIRE package. That
package requires Python older than 3.13, and positronic runs on 3.13. See [NOTICE](NOTICE) for the source and the
licence.

## Run

Serve the programs in a directory. `--programs_dir` is [`examples`](examples) by default:

```bash
uv run --locked python -m positronic.vendors.aspire.server \
  --programs_dir=positronic/vendors/aspire/examples --websocket.served_address.port=8000
```

Each session names its program with the session parameter `program`, a path relative to `--programs_dir`. The
server reads the file when the session opens, so an edit applies to the next session. A session that names no file in
the directory fails at connect. Run a program on the bimanual YAM in the ABC simulator:

```bash
uv run --locked positronic eval run --eval=.sim.amazon_abc.put_bottles --eval.timeout=60 \
  --policy=.remote --policy.address.port=8000 --policy.address.query=program=lift_and_return.py \
  --output_dir=~/datasets/aspire/abc
```

[`examples/lift_and_return.py`](examples/lift_and_return.py) lifts each hand by 5 cm, closes and opens the grippers,
returns the hands together, and homes each arm. It reads no camera. It runs on a rig with one arm or two.

The server serves a two-arm rig by default. For a rig whose channels carry no arm name (`robot_state.q`, `grip`,
`robot_command`, `target_grip`), start it with `--settings.arms='[None]'`:

```bash
uv run --locked python -m positronic.vendors.aspire.server --settings.arms='[None]' \
  --calibration=path/to/calibration.json --websocket.served_address.port=8000
```

## How a call runs

The program runs in a thread of its own on the server. A call that moves the robot puts its target in a queue and
blocks. Each inference request reads one observation, decides whether the move in flight ended, wakes the program when
it did, and answers the current targets as a chunk. The client plays the chunk at `--settings.fps` and asks again when
the chunk ends. A chunk lasts at least `--settings.round_s`. An answer waits up to `--settings.think_timeout_s` of wall
time for the program's next call; after that it holds the targets for one chunk.

The bridge decides that a move ended, from the measured state:

| End | When |
| --- | --- |
| `arrived` | Each arm the call moves is within the bridge's tolerances of its target: `position_tolerance_m`, `rotation_tolerance_rad`, `joint_tolerance_rad`, `grip_tolerance` |
| `stopped` | The joints and grippers of those arms stay within `still_joint_rad` and `still_grip` for `still_rounds` chunks, short of the target. A grasp that stalls on an object ends so |
| `timeout` | `call_timeout_s` passes after the call's chunk ends. The bridge counts time in the chunks it answered, so a slow simulator times out as a fast one does |
| `period` | `step` ends after one chunk |

A result keeps three facts apart:

- `diagnostics['accepted']`: the bridge took the target. A refused call has `ok=False`, `steps_executed=0` and moves
  nothing.
- `diagnostics['arrived']`: the move ended as `arrived`. `diagnostics['move_end']` names the end.
- `ok`: the final state meets the call, as ASPIRE's CapApi judges it. A pose move must be within the call's
  `tolerance` in metres and within 10 degrees. A joint move must be within `tolerance` in radians. A gripper call must
  arrive. A failed move carries `timeout` or `controller_failed`.

`max_steps` and `strategy` are accepted and not used: the bridge's timeout bounds each call, and the rig solves IK.

## Mapping

| CapApi | Rig |
| --- | --- |
| `move_to_pose`, a pose in `move_synchronized` | `CartesianPosition` on `robot_command.{arm}` |
| `move_to_joints`, `go_home`, `step`, `execute_trajectory`, joints in `move_synchronized` | `JointPosition` on `robot_command.{arm}` |
| `set_gripper(s)`, `open_gripper`, `close_gripper` | `target_grip.{arm}`, as `1 - position`: ASPIRE's 1 is open, positronic's 1 is closed |
| The program ends | `end` with `complete` on every later answer |
| The program raises, or does not compile | `end` with `give_up`; the server logs the traceback and records it |

An arm that a call does not move keeps its last target. At the first observation each arm holds its measured joints,
and `go_home` returns an arm to those joints. `execute_trajectory` plays its waypoints at the chunk rate and refuses a
trajectory that does not start within 0.02 rad of the measured joints (`stale_plan`).

`--settings.arms` names the arms of the rig as its channels do. The default is `left` and `right`.
`--settings.aliases` maps ASPIRE's `primary` and `secondary` to them; the default maps `primary` to `right`. On a
one-arm rig (`--settings.arms='[None]'`) the arm is `primary`, and the channels are those of the table without
`.{arm}`. An arm has the joint count that its first observation holds: 6 for a YAM, 7 for a Franka. The bridge refuses
a joint target of a different length (`invalid_request`). `--settings.embodiment` (`yam_real` by default) is the
embodiment name of each state and command in a program.

Poses are in one frame, `--settings.base_frame` (`world` by default): the frame of the rig's measured hand poses, at
the `gripper` site. The server states that site in its `ee_frames`.

`get_task_context` carries the task instruction as `language` and `task_name`.

## Camera views

`get_observation` returns the view of each `image.*` camera, named without the prefix, for example `exterior`. A view
always holds `rgb`. It holds more when the rig and the server give more:

| Field | Source |
| --- | --- |
| `depth_m` | The observation's `depth.{camera}`, beside `image.{camera}` and with its height and width. The bridge reads `uint16` millimetres or floating metres. The view holds `float32` metres, with 0 where the camera measured no depth |
| `intrinsics`, `camera_pose` | The calibration file, `--calibration` |

No camera driver in positronic sends `depth.{camera}`. With no depth and no calibration, a view holds `rgb` only.

The calibration file is a JSON object. It maps a camera name to the camera's matrix and the pose of its optical frame:
x to the right of the image, y down, z forward. A pose is `x, y, z, qw, qx, qy, qz`, as `robot_state.ee_pose` is.

```json
{
  "exterior": {"intrinsics": [[525.0, 0.0, 640.0], [0.0, 525.0, 360.0], [0.0, 0.0, 1.0]],
               "pose": [0.9, 0.4, 0.6, 0.0, 0.92388, 0.38268, 0.0]},
  "wrist": {"intrinsics": [[365.0, 0.0, 640.0], [0.0, 365.0, 360.0], [0.0, 0.0, 1.0]],
            "pose": [0.0, -0.06, 0.03, 1.0, 0.0, 0.0, 0.0], "hand": "primary"}
}
```

The pose of a camera with no `hand` is in the base frame. The pose of a camera with `hand` is in the hand frame of
that arm, as a program names the arm (`primary` names the right arm of the default two-arm rig). The bridge then gives `camera_pose` as the measured hand pose times this pose,
at each observation. The hand frame is the frame of `robot_state.ee_pose`. The server refuses a `hand` that names no arm when it
starts. It states the calibration in its metadata, under `calibration`.

## Perception

The bridge holds no perception model. The server takes providers, which are objects with the methods of ASPIRE's
provider protocols (`SegmentationProvider` and `GraspProvider` in [`contracts.py`](contracts.py)):

```bash
uv run --locked python -m positronic.vendors.aspire.server \
  --segmentation=@my_package.providers.segmentation --grasps=@my_package.providers.grasps
```

`--segmentation` names a configuronic config of one segmentation provider. `--grasps` names a mapping from a backend
name, which is the `backend` argument of `generate_grasps`, to the config of a grasp provider. All sessions share the
providers, so a provider can get calls from more than one thread. A program's thread calls a provider, and the robot
holds its targets during the call.

| Call | Needs | What it does |
| --- | --- | --- |
| `segment_text`, `segment_points` | A segmentation provider | Asks the provider about the view of one camera |
| `mask_to_point_cloud` | A view with depth and intrinsics | Gives the points of a mask in the base frame, or in the camera frame when `target_frame` is the camera name |
| `crop_point_cloud`, `estimate_geometry` | Nothing | Crops a point cloud to bounds; gives the oriented box of its points |
| `localize_object` | A segmentation provider | Boxes the points of the provider's first segmentation. It reads `--settings.default_camera` (`exterior`) when the call names no camera |
| `generate_grasps` | A grasp provider for the call's `backend` | Gives the provider the points of the mask and the points of the full view, in the base frame |
| `select_grasp` | Nothing | Gives the grasp with the highest score among those whose z axis is within 37 degrees of the frame's -z axis (`top_down`, the default), or among all (`highest_score`) |

A call with no provider raises `NotImplementedError`. A provider that raises gives the program a result with
`ok=False` and its error: an `ApiError` as the provider raised it, or `provider_error` for each other exception. A
view that cannot give points gives `point_cloud_failed` from `localize_object` and `generate_grasps`, and raises
`ValueError` from `mask_to_point_cloud`. A segmentation must be in the frame of its view, which is the camera name:
the bridge gives `provider_error` for a segmentation in another frame.

Planning calls (`solve_ik`, `plan_motion`, `plan_synchronized_motion`) raise `NotImplementedError`.

## Program rules

A program may import the modules that `--settings.imports` names: `math` only by default. It sees ASPIRE's small set
of builtins. `--settings.imports='[math,numpy]'` adds `numpy`, which can read and write files. These rules keep a
program to ASPIRE's contract; they are not a sandbox. Run a program only from a source you trust.

In a program, `ArmCommand`, `Trajectory` and `SynchronizedTrajectory` make values with the embodiment of
`--settings.embodiment`. They are functions, so `isinstance` does not take them.

## Records

The server writes `<sessions_dir>/<session_id>.json` when a program ends and again when its session ends. The record
holds the program name, its source and SHA-256, the `end`, the traceback of a program that raised, and `calls`: one
entry for each CapApi call the program made. A session that ended before its program has no `end`.
`--sessions_dir` is `aspire_sessions` under the working directory by default.

An entry of `calls` holds:

| Field | Value |
| --- | --- |
| `name`, `args`, `kwargs` | The call as the program made it |
| `result` | The value the call returned, without camera images. A move result names its end in `diagnostics['move_end']` |
| `raised` | The exception the call raised, for example `NotImplementedError` |
| `started_wall_s`, `ended_wall_s` | Server wall time, in seconds from the program's start |
| `started_robot_s`, `ended_robot_s` | Robot time, in seconds of the chunks the bridge answered |

In `args`, `kwargs` and `result`, an array with two or more dimensions, such as a mask, is its `shape`, its `dtype` and
its `nonzero` count. A point cloud is its `frame` and its `point_count`.

The episode records the program name, its SHA-256 and the record's file name (`session_record`) under
`inference.policy.server`.

## Tests

```bash
uv run --locked pytest positronic/vendors/aspire/tests
```

The tests drive the bridge with fake observation streams of a two-arm rig and of a one-arm rig, and with fake
providers. They need no simulator and no perception model.
