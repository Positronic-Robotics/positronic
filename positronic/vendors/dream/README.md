# Dream-1 in Positronic

> **Status: Prototype** – Dream-1 runs in the ABC sim of the bimanual YAM. It has not run on a real robot. Read
> [NOTICE](./NOTICE) before you use it: the terms of DreamAPI limit what you may do with it.

## What is Dream-1?

[Dream-1](https://dreamlabs.ai/research/dream-1/) is a bimanual policy of Dream Labs, trained on i2rt YAM arms. It
runs only behind DreamAPI. Dream Labs does not release the weights. Each call sends the last 9 frames of 3 cameras
(0.2 s apart), the joints and grippers of both arms, and a prompt. Each reply is a chunk of 48 rows at 20 Hz: per arm,
6 joints and the gripper opening.

## How it runs

`server.py` serves Dream-1 over positronic's session protocol, and calls DreamAPI with `dreamlabs-client`:

- The client stacks 9 frames of each camera, 0.2 s apart (`TemporalStack`), each bounded to 320 pixels wide. It asks
  for the next chunk before the current one ends, by the longest of the last 5 round trips (at most
  `--pipeline.max_lead_s`, 1.6 s), and sends the rest of its plan. The next chunk starts when the current one ends,
  so the server drops the rows of the answer that the client plays first, and the limiter starts from the last row
  sent.
- The server paces the chunks as `dreamlabs-client` does. In the client's `async` mode (a median round trip of
  DreamAPI of at most 700 ms) a chunk keeps the next 30 rows; in `sync` mode it keeps every row.
- The server sends the YAM station observation of the Dream SDK (`state.arm_l.joint_pos`, `image_window.top`, ...).
  DreamAPI computes each hand's pose from the joints, so no IK runs here.
- The server blurs every face before a frame leaves it, and limits each chunk (`limits.py`): the hand moves at most
  0.6 m/s, each gripper changes by at most 0.05 per row, and each joint moves at most 0.5 rad/s for the first 2 s.
  These are the Default Limits of the Dream SDK station. Each joint also stays 0.03 rad inside its range, as the
  station's tracker clips its reference, because Dream commands joints 2 and 3 past their 0 stop.
- Each session first sends a canary: a fixed observation. A change of its hash means that the model changed. DreamAPI
  gives no model version, and `dreamlabs/dream-1-beta` can change at any time.
- Each session writes `<sessions_dir>/<session_id>.json` (the canary, and the trace id, timing and limits of each
  call) and `<session_id>.npz` (`answered`: each chunk as DreamAPI answered it; `limited`: each chunk as the limiter
  sent it, NaN in each row that the server did not send). A replan drops the rest of a chunk, so `limited` can hold
  rows that the client did not run. Dream Labs can find a call from its trace id.
- `upload_probe.py` times the frame path of one call on this machine at a given upload size: the resize and JPEG of
  the client, and the decode, cut, face blur and JPEG of the server.

## Run in simulation

Get a DreamAPI key at [platform.dreamlabs.ai](https://platform.dreamlabs.ai). Put it in `DREAM_API_KEY`, or in
`~/.dreamlabs/credentials` as the line `DREAM_API_KEY=<key>` with mode 600. Do not put it in arguments or configs.

```bash
# Serve Dream-1 on 127.0.0.1:8000. A cold model takes up to an hour to start, and the server waits for it.
uv run --locked --extra dream python -m positronic.vendors.dream.server yam_bimanual \
  --websocket.served_address.host=127.0.0.1 --model.sessions_dir=~/dream_sessions

# Run an ABC task against it. Name the same extras in both commands: `uv run` removes the extras it is not given.
uv run --locked --extra dream positronic eval run --eval=.sim.amazon_abc.put_bottles \
  --policy=.remote --policy.address.host=localhost --policy.address.port=8000 \
  --charge_inference_time=False --output_dir=~/datasets/dream/abc
```

The server binds `0.0.0.0` unless told otherwise, and with no `AUTH_TOKEN` it serves any client. Each request from a
client becomes a billed DreamAPI call on your key, so bind `127.0.0.1` when the client runs on the same machine. A
server that a robot reaches from another machine needs `AUTH_TOKEN` and a private network, such as a tailnet (see
[Authentication](../../offboard/README.md#authentication)).

A round trip to DreamAPI takes 0.4 to 4 s, so a sim run pauses sim time during each call
(`--charge_inference_time=False`). On a machine with no GPU, set `MUJOCO_GL=osmesa`. DreamAPI ended a session that
waited 39 s between calls in a test. The server starts a new session after `--model.session_idle_s` seconds
of no calls.

DreamAPI checks each instruction against the scene, and refuses one that it finds unsafe: `DA-4000 malformed`,
"unsafe instruction in this scene". The session then fails. In ABC, `turn_mug_right_side_up` got this refusal.

## Known gaps

- The hand speed limit scales a joint step once. A step can move the hand a little more than the limit.
- The limits do not include joint speed, joint torque or a workspace boundary. A real robot needs them, with an
  emergency stop that does not depend on this server.
