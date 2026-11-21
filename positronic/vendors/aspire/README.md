# ASPIRE CapApi programs

[ASPIRE](https://github.com/NVlabs/ASPIRE) agents write Python programs against its CapApi: `move_to_pose`,
`set_gripper`, `get_robot_state` and other calls. This vendor runs such a program as an ordinary remote policy. The rig
sends observations and plays the action chunks that the server answers. The program never drives a motor. The rig's
driver solves IK and limits speed.

The vendor holds a port of the ASPIRE value types (`contracts.py`) and does not depend on the ASPIRE package. That
package requires Python older than 3.13, and positronic runs on 3.13. See [NOTICE](NOTICE) for the source and the
licence.

## Run

Serve one program. Each session runs a new copy of it:

```bash
uv run --locked python -m positronic.vendors.aspire.server \
  --program=positronic/vendors/aspire/examples/lift_and_return.py --websocket.served_address.port=8000
```

Run it on the bimanual YAM in the ABC simulator:

```bash
uv run --locked positronic eval run --eval=.sim.amazon_abc.put_bottles --eval.timeout=60 \
  --policy=.remote --policy.address.port=8000 --output_dir=~/datasets/aspire/abc
```

[`examples/lift_and_return.py`](examples/lift_and_return.py) lifts each hand by 5 cm, closes and opens both grippers,
returns both hands together, and homes both arms. It reads no camera.

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
| The program raises | `end` with `give_up`; the server logs the traceback and records it |

An arm that a call does not move keeps its last target. At the first observation each arm holds its measured joints,
and `go_home` returns an arm to those joints. `execute_trajectory` plays its waypoints at the chunk rate and refuses a
trajectory that does not start within 0.02 rad of the measured joints (`stale_plan`).

Arms are `left` and `right`. `--settings.aliases` maps ASPIRE's `primary` and `secondary` to them; the default maps
`primary` to `right`. Poses are in one frame, `--settings.base_frame` (`world` by default): the frame of the rig's
measured hand poses, at the `gripper` site. The server states that site in its `ee_frames`.

`get_observation` returns the RGB view of each `image.*` camera, named without the prefix, for example `exterior`. A
view has no depth, intrinsics or pose. `get_task_context` carries the task instruction as `language` and `task_name`.

Perception and planning calls (`segment_text`, `localize_object`, `plan_motion`, `solve_ik` and the others) raise
`NotImplementedError`. A program may import `math` only, and it sees ASPIRE's small set of builtins. These rules keep a
program to ASPIRE's contract; they are not a sandbox. Run a program only from a source you trust.

## Records

The server writes `<sessions_dir>/<session_id>.json` when a program ends and again when its session ends. The record
holds the SHA-256 of the program, the `end`, and the traceback of a program that raised. A session that ended before
its program has no `end`. `--sessions_dir` is `aspire_sessions` under the working directory by default. The episode
records the program name and its SHA-256 under `inference.policy.server`.

## Tests

```bash
uv run --locked pytest positronic/vendors/aspire/tests
```

The tests drive the bridge with a fake two-arm observation stream. They need no simulator.
