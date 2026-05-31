# Codec Guide

A trained model expects its inputs in one particular shape: end-effector pose or joint angles, absolute targets or deltas, quaternions or 6D rotations, 224×224 or 512×512 images. Your recordings know none of that — they hold the raw robot state and the commands that were sent. A **codec** is the transformation between the two: it projects your raw data into exactly the state and action space a given model expects.

This is what lets you **record once and run any model**. The same raw episode can train and serve a LeRobot policy in end-effector space, a GR00T policy in 6D-rotation joint space, and an OpenPI policy in absolute-position space — you only swap the codec. No re-recording datasets per model, and no throwing data away when you switch formats. See the [Dataset Library](../positronic/dataset/README.md) for how the raw data is stored and transformed lazily.

## One codec, both directions

The interesting part is that a codec defines its transformation for **both** training and inference, in a single place:

- **Training** — `training_encoder` derives the model's columns from a recorded episode, lazily, across the whole dataset: the observation features the model will see and the action labels it should learn.
- **Inference** — `encode()` turns a live raw observation into the model's input, and `decode()` turns the model's output back into a robot command.

Because the same codec owns both paths, the data a model trains on and the data it sees at inference cannot drift apart — they are produced by the same code. (When they *do* mismatch, it is almost always because training and inference were pointed at different codecs — see [Matching training and inference](#matching-training-and-inference).)

That dual structure is also why codecs **compose**. Each codec is a small piece that owns both directions, and two operators combine them so that the training transform and the inference transform compose together in lock-step:

- `&` (parallel): both sides see the same input and their outputs are merged. The standard pairing is `observation & action` — one encodes what the model sees, the other what it predicts.
- `|` (sequential): the left codec transforms the data before the right one sees it. Use it for steps that must run first — binarizing the grip signal, or replacing an end-effector target with a joint target (below) — ahead of the observation and action encoders.

## Observation and action encoding

The two codecs that do the real work are the **observation encoder** and the **action decoder**.

**Observation encoding** chooses which raw fields the model sees, and in what form. `ObservationCodec` ([`positronic/policy/observation.py`](../positronic/policy/observation.py)) is configured with state vectors to assemble (e.g. concatenate `robot_state.ee_pose` + `grip`) and images to resize. The same configuration builds the training columns and encodes the live observation, so the two match by construction.

**Action encoding** chooses what the model predicts and how that maps back to a robot command. This is where the action-space decisions live. Some real examples from [`positronic/policy/action.py`](../positronic/policy/action.py):

- **Absolute end-effector** (`AbsolutePositionAction`): the model predicts a target pose `[translation, rotation, grip]`. In training the label is the commanded EE pose in the chosen rotation representation; at inference `decode` turns the predicted vector into a `CartesianPosition` command. The model reasons in EE space, the robot is driven in EE space.
- **Absolute joints** (`AbsoluteJointsAction`): the model predicts joint angles `[q…, grip]` directly, and `decode` produces a `JointPosition` command — no inverse kinematics at runtime.
- **End-effector → joint targets via IK** (`IKJointsAction`): you recorded the robot in *end-effector* space (`robot_commands.pose`) but want a *joint-space* model. `IKJointsAction` runs inverse kinematics over each episode — seeded by the recorded `robot_state.q` — to compute the joint targets that reach those poses, and swaps them in as the action labels. You compose it with `|` ahead of `AbsoluteJointsAction`, which decodes the model's joint output at inference; `IKJointsAction` itself is then a pass-through, since the conversion only had to happen once, when building the training set. This is the sharpest illustration of the whole idea: the *same* end-effector recordings train either an EE-space or a joint-space model, just by changing the codec.
- **Relative / delta** (`RelativePositionAction`): the model predicts a displacement from the current pose; `decode` reads the live `robot_state.ee_pose` from its context and adds the delta. A smaller action space for relative policies.

### Commanded vs observed targets

By default, action codecs label actions with the **commanded** targets (`robot_commands.pose`, `target_grip`) — "what the controller was told to do." The `_traj` variants instead use the **actual** robot trajectory (`robot_state.ee_pose`, `grip`) — "what the robot actually did" — and binarize the observed grip (continuous → open/close), since the model should learn a discrete grip. Same raw data, two different notions of the action label.

## Timing codecs

A returned trajectory also needs timing — *when* each action runs. That is handled by an optional extra stage, composed to the left of the obs/action codecs (see `compose` in [`positronic/cfg/codecs.py`](../positronic/cfg/codecs.py); implementation in [`positronic/policy/codec.py`](../positronic/policy/codec.py)):

| Codec | Signature | Effect |
|-------|-----------|--------|
| `ActionTimestamp` | `ActionTimestamp(fps=...)` (keyword-only) | Stamps each decoded action with a relative `timestamp = i / fps` (seconds from the start of the trajectory, starting at 0). At training time surfaces `action_fps` as transform metadata. (`codec.py:183`) |
| `ActionHorizon` | `ActionHorizon(horizon_sec)` (positional) | Drops decoded actions whose relative `timestamp` is `>= horizon_sec`. Single (untimestamped) actions pass through. At training time surfaces `action_horizon_sec`. (`codec.py:217`) |
| `ActionTiming` | `ActionTiming(fps=..., horizon_sec=None)` | Factory: returns `ActionTimestamp(fps=fps) \| ActionHorizon(horizon_sec)` when `horizon_sec` is set, otherwise just `ActionTimestamp(fps=fps)`. (`codec.py:247`) |

These timestamps are **relative** — seconds from the start of the trajectory. The model and codecs never see wall-clock time; the real-time client anchors each offset to its own clock the moment inference returns (`now + timestamp`), so execution starts at inference-finish and the round-trip latency is absorbed. Stamping a per-action timestamp rather than a fixed rate is deliberate: it lets non-uniform timings and client-side scheduling strategies share one wire format. See [How inference works](connect-your-model.md#how-inference-works) for the full reasoning.

## Writing custom codecs

Subclass `positronic.policy.codec.Codec` and implement `encode()` and/or `_decode_single()`. The base class returns `{}` from both — observation codecs override `encode()`, action codecs override `_decode_single()`. Middleware codecs that pass data through must explicitly `return data` (e.g. `BinarizeGripTraining`, a pure pass-through at decode that only binarizes via its `training_encoder`); middleware that transforms decoded actions modifies and returns `data` instead (e.g. `BinarizeGripInference`, which thresholds `target_grip` in `_decode_single`). Compose observation and action codecs with `&`, chain middleware with `|`. See the vendor codec files below for reference patterns.

## Codec catalog by vendor

These are the ready-made codecs each vendor ships. The standard composition is `[ActionHorizon] | ActionTimestamp | [BinarizeGrip…] | observation & action` (the bracketed stages are controlled by `compose`'s `horizon` and `binarize_grip=` arguments).

### LeRobot (ACT — 0.3.3)

See [`positronic/vendors/lerobot_0_3_3/codecs.py`](../positronic/vendors/lerobot_0_3_3/codecs.py).

| Codec | Observation | Action |
|-------|-------------|--------|
| `ee` | EE pose (7D quat) + grip + images (224x224) | Absolute EE position (7D quat) + grip |
| `joints` | Joint positions (7D) + grip + images | Absolute EE position (7D quat) + grip |
| `ee_traj` | EE pose (7D quat) + grip + images (224x224) | Absolute EE trajectory (7D quat) + grip (binarized) |
| `joints_traj` | Joint positions (7D) + grip + images | Absolute joint trajectory (7D) + grip (binarized) |

```bash
cd docker && docker compose run --rm lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task
```

### LeRobot (SmolVLA — 0.4.x)

See [`positronic/vendors/lerobot/codecs.py`](../positronic/vendors/lerobot/codecs.py).

| Codec | Observation | Action |
|-------|-------------|--------|
| `ee` | EE pose (7D quat) + grip + images (512x512) | Absolute EE position (7D quat) + grip |
| `joints` | Joint positions (7D) + grip + images (512x512) | Absolute EE position (7D quat) + grip |

```bash
cd docker && docker compose run --rm lerobot-convert convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task
```

### GR00T

See [`positronic/vendors/gr00t/codecs.py`](../positronic/vendors/gr00t/codecs.py).

| Codec | Observation | Action | Modality Configs |
|-------|-------------|--------|------------------|
| `ee_quat` | EE pose (quat) + grip + images (224x224) | Absolute EE position (quat) + grip | `ee`, `ee_rel` |
| `ee_rot6d` | EE pose (rot6d) + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d`, `ee_rot6d_rel` |
| `ee_quat_joints` | EE pose + joints + grip + images | Absolute EE position + grip | `ee_q`, `ee_q_rel` |
| `ee_rot6d_joints` | EE pose (rot6d) + joints + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d_q`, `ee_rot6d_q_rel` |
| `ee_quat_traj` | EE pose (quat) + grip + images | Absolute EE trajectory (quat) + grip (binarized) | `ee`, `ee_rel` |
| `ee_rot6d_traj` | EE pose (rot6d) + grip + images | Absolute EE trajectory (rot6d) + grip (binarized) | `ee_rot6d`, `ee_rot6d_rel` |
| `ee_quat_joints_traj` | EE pose + joints + grip + images | Absolute EE trajectory + grip (binarized) | `ee_q`, `ee_q_rel` |
| `ee_rot6d_joints_traj` | EE pose (rot6d) + joints + grip + images | Absolute EE trajectory (rot6d) + grip (binarized) | `ee_rot6d_q`, `ee_rot6d_q_rel` |
| `joints_traj` | Joints + grip + images (no EE pose) | Absolute joint trajectory + grip (binarized) | — |

The codec must match the modality config during training.

```bash
# Convert with codec
cd docker && docker compose run --rm lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
  --output_dir=~/datasets/groot/my_task

# Train with matching modality
cd docker && docker compose run --rm groot-train \
  --modality_config=ee_rot6d_q \
  --input_path=~/datasets/groot/my_task
```

### OpenPI

See [`positronic/vendors/openpi/codecs.py`](../positronic/vendors/openpi/codecs.py).

| Codec | Observation | Action |
|-------|-------------|--------|
| `ee` | EE pose (7D quat) + grip + images (224x224) | Absolute EE position (7D) |
| `ee_joints` | EE pose + joints (7D) + grip + images | Absolute EE position (7D) |
| `ee_traj` | EE pose (7D quat) + grip + images (224x224) | Absolute EE trajectory (7D) + grip (binarized) |
| `ee_joints_traj` | EE pose + joints (7D) + grip + images | Absolute EE trajectory (7D) + grip (binarized) |
| `joints_traj` | Joints (7D) + grip + images | Absolute joint trajectory (7D) + grip (binarized) |
| `droid` | Joint positions (7D) + grip + images | Joint delta (velocity) |

`droid` is inference-only, for pretrained DROID models (not for training).

```bash
cd docker && docker compose run --rm lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.openpi.codecs.ee \
  --output_dir=~/datasets/openpi/my_task
```

### Choosing a codec

The main decision is the observation space: **end-effector pose only** vs **end-effector + joint positions**. Joint feedback can help learning but isn't always needed. Action spaces (absolute position, joint targets, deltas) are supported but vary by vendor — check the tables above.

### Matching training and inference

A "Shape mismatch" or "Feature mismatch" at inference almost always means the inference codec differs from the one used to convert the training data. They must be identical:

```bash
# Training (ACT — 0.3.3)
cd docker && docker compose run --rm lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee

# Inference must match
cd docker && docker compose run --rm --service-ports lerobot-0_3_3-server \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee
```

## See Also

- [Connect Your Model](connect-your-model.md) – the inference API and where codecs sit in it
- [Dataset Library README](../positronic/dataset/README.md) – raw storage and transforms
- [Training Workflow](training-workflow.md) – using codecs in the pipeline
- Vendor docs: [LeRobot ACT](../positronic/vendors/lerobot_0_3_3/README.md) | [SmolVLA](../positronic/vendors/lerobot/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [OpenPI](../positronic/vendors/openpi/README.md)
