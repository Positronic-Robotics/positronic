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

- The client stacks 9 frames of each camera, 0.2 s apart (`TemporalStack`), and plays each chunk whole.
- The server sends the YAM station observation of the Dream SDK (`state.arm_l.joint_pos`, `image_window.top`, ...).
  DreamAPI computes each hand's pose from the joints, so no IK runs here.
- The server blurs every face before a frame leaves it, and limits each chunk (`limits.py`): the hand moves at most
  0.6 m/s, each gripper changes by at most 0.05 per row, and each joint moves at most 0.5 rad/s for the first 2 s.
  These are the Default Limits of the Dream SDK station. Each joint also stays 0.03 rad inside its range, as the
  station's tracker clips its reference, because Dream commands joints 2 and 3 past their 0 stop.
- Each session first sends a canary: a fixed observation. A change of its hash means that the model changed. DreamAPI
  gives no model version, and `dreamlabs/dream-1-beta` can change at any time.
- Each session writes `<sessions_dir>/<session_id>.json` (the canary, and the trace id, timing and limits of each
  call) and `<session_id>.npz` (each chunk as DreamAPI answered it and as the client played it). Dream Labs can find
  a call from its trace id.

## Run in simulation

Get a DreamAPI key at [platform.dreamlabs.ai](https://platform.dreamlabs.ai). Put it in `DREAM_API_KEY`, or in
`~/.dreamlabs/credentials` as the line `DREAM_API_KEY=<key>` with mode 600. Do not put it in arguments or configs.

```bash
# Serve Dream-1 on :8000. A cold model takes up to an hour to start, and the server waits for it.
uv run --locked --extra dream python -m positronic.vendors.dream.server yam_bimanual \
  --model.sessions_dir=~/dream_sessions

# Run an ABC task against it. Name the same extras in both commands: `uv run` removes the extras it is not given.
uv run --locked --extra dream positronic eval run --eval=.sim.amazon_abc.put_bottles \
  --policy=.remote --policy.address.host=localhost --policy.address.port=8000 \
  --charge_inference_time=False --output_dir=~/datasets/dream/abc
```

A round trip to DreamAPI takes 0.4 to 4 s, so a sim run pauses sim time during each call
(`--charge_inference_time=False`). On a machine with no GPU, set `MUJOCO_GL=osmesa`. DreamAPI ended a session that
waited 39 s between calls in a test. The server starts a new session after `--model.session_idle_s` seconds
of no calls.

DreamAPI checks each instruction against the scene, and refuses one that it finds unsafe: `DA-4000 malformed`,
"unsafe instruction in this scene". The session then fails. In ABC, `turn_mug_right_side_up` got this refusal.

## Known gaps

- The hand speed limit scales a joint step once. A step can move the hand a little more than the limit.
- The ramp counts each chunk as played whole. A client that plays fewer rows ends the ramp early.
- The limits do not include joint speed, joint torque or a workspace boundary. A real robot needs them, with an
  emergency stop that does not depend on this server.
