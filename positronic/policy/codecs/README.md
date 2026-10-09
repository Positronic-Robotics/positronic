# Codec Guide

A **codec** converts recorded robot data or live observations into model inputs, and decodes model outputs into robot commands. Choose codecs to match the checkpoint's state and action representations, image preparation, gripper convention and end-effector frame.

The same recordings can supply different model formats when they contain the required signals. See the [Dataset Library](../../dataset/README.md) for how raw data is stored and transformed lazily.

Import reusable codecs from `positronic.policy.codecs` or their modules below.
The package root exposes public names; implementations and their tests are grouped by responsibility:

| Module | Responsibility |
|--------|----------------|
| [`base`](base.py) | `Codec` and sequential/parallel composition |
| [`observation`](observation.py) | Field names, state vectors and model input assembly |
| [`action`](action.py) | Prediction decoding, training labels and command control modes |
| [`geometry`](geometry.py) | Pose representations and end-effector frames |
| [`gripper`](gripper.py) | Grip thresholds and conventions |
| [`image`](image.py) | Image size limits and JPEG encoding |
| [`metadata`](metadata.py) | Metadata attachment and dataset feature descriptors |

Codec tests live in [`tests/`](tests/). Model-specific recipes compose these operations under
[`positronic/vendors/`](../../vendors/); configuration presets live in
[`positronic/cfg/codecs.py`](../../cfg/codecs.py).

Use the [reusable codec catalog](#reusable-codec-catalog) to assemble a recipe, or the
[vendor recipes](#codec-catalog-by-vendor) to choose a configured pipeline.

## One codec, both directions

A codec can define transformations for training and inference in one place:

- **Training** — `training_encoder` derives the model's columns from a recorded episode, lazily, across the whole dataset: the observation features the model will see and the action labels it should learn.
- **Inference** — `encode()` turns a live raw observation into the model's input, and `decode()` turns the model's output back into a robot command.

Shared conversion settings keep training and inference consistent. Some codecs apply only to one path; the catalog below identifies them. Keep model input and action conventions aligned when selecting recipes — see [Matching training and inference](#matching-training-and-inference).

That dual structure is also why codecs **compose**. Each codec is a small piece that owns both directions, and two operators combine them so that the training transform and the inference transform compose together in lock-step:

- `&` (parallel): both sides see the same input and their outputs are merged. The standard pairing is `observation & action` — one encodes what the model sees, the other what it predicts.
- `|` (sequential): the left codec transforms the data before the right one sees it. Use it for steps that must run first — binarizing the grip signal, or replacing an end-effector target with a joint target (below) — ahead of the observation and action encoders.

Sequential decoding runs in reverse order. For example, `BinarizeGripInference() | action`
decodes the model output with `action` first, then thresholds the decoded grip.

## Reusable codec catalog

These are the public codec classes exported by this package. Each name links to its implementation.

### Composition and metadata

| Codec | What it does | When to use it |
|-------|--------------|----------------|
| [`Codec`](base.py) | Base class for observation encoding, action decoding, training transforms and composition. | Subclass it when a conversion is not covered by the existing codecs. |
| [`Metadata`](metadata.py) | Attaches metadata to the codec and its training transform; passes data through unchanged. | Add shared metadata to a composed recipe without another data conversion. |

### Observations

| Codec | What it does | When to use it |
|-------|--------------|----------------|
| [`ObservationCodec`](observation.py) | Concatenates state signals, resizes and pads RGB images, and carries the task prompt. Shares state/image settings between training and inference. | Build named state vectors and image fields from raw robot observations. |
| [`RenameObservationFields`](observation.py) | Renames literal top-level inference fields. Training columns and decoded actions pass through. | The model's inference input names differ from the training column names. |

### Actions

| Codec | What it does | When to use it |
|-------|--------------|----------------|
| [`AbsolutePositionAction`](action.py) | Builds pose/grip training labels and decodes predictions into `CartesianPosition` commands plus grip. | The model predicts absolute end-effector poses in a chosen rotation representation. |
| [`AbsoluteJointsAction`](action.py) | Builds joint/grip training labels and decodes predictions into `JointPosition` commands plus grip. | The model predicts absolute joint positions. |
| [`IKJointsAction`](action.py) | Training: replaces pose targets with joint targets through inverse kinematics. Inference: passes through. | Train a joint-position model from recorded pose targets; compose before `AbsoluteJointsAction`. |
| [`JointDeltaAction`](action.py) | Inference: clips and scales normalized DROID joint velocities into `JointDelta` commands, and thresholds grip. | Serve a checkpoint using the DROID joint-delta action convention. |
| [`SetControlMode`](action.py) | Inference: stamps a supplied control mode on each decoded arm command. | Require explicit impedance or position control; compose to the left of the action decoder. |

### Geometry

| Codec | What it does | When to use it |
|-------|--------------|----------------|
| [`ChangeEEFrame`](geometry.py) | Moves observation and recorded poses into a tool frame, records the frame, and converts decoded commands back. | The checkpoint uses a different physical end-effector frame from the robot's `default` frame. |
| [`ConvertPose`](geometry.py) | Converts selected observation/episode pose vectors to float32, a chosen rotation representation and a fixed rotation offset. Preserves translation and frame metadata; decoded actions pass through. | Match quaternion, Euler or 6D rotation conventions expected by a model. |

### Gripper

| Codec | What it does | When to use it |
|-------|--------------|----------------|
| [`BinarizeGripTraining`](gripper.py) | Training: thresholds selected signals to 0 or 1. Inference: passes through. | Recordings contain continuous grip values but the model should learn binary grip labels or inputs. |
| [`BinarizeGripInference`](gripper.py) | Inference: thresholds the decoded grip command. Training: passes through. | The model predicts continuous grip values but execution needs open/closed commands. |
| [`FlipGrip`](gripper.py) | Inference: maps observed grip and decoded target grip to `1 - value`. Training: passes through. | Serve checkpoints trained with `1 = open` on Positronic's `1 = closed` convention. |

### Images

| Codec | What it does | When to use it |
|-------|--------------|----------------|
| [`RestrictImageSize`](image.py) | Inference: shrinks nested RGB images and frame stacks to fit bounds, preserving aspect ratio. Does not upscale; training encoding is unsupported. | Cap image dimensions before sending observations over the network. |
| [`EncodeImages`](image.py) | Inference: JPEG-encodes images recursively, selecting uint8 RGB arrays automatically. Supports explicit paths and JPEG quality. | Reduce network payload size with lossy image compression after preparing the model inputs. |

Image transport codecs belong in the serving stack. Use `ObservationCodec` for image preparation
shared with training. `IKJointsAction` and `SetControlMode` require Python construction; they do not
provide JSON component descriptions.

The [metadata module](metadata.py) also provides `lerobot_vector`, `lerobot_image` and
`lerobot_action` to describe dataset features. These helpers return metadata dictionaries, not codecs.

## Observation and action encoding

The two codecs that do the real work are the **observation encoder** and the **action decoder**.

**Observation encoding** chooses which raw fields the model sees, and in what form. `ObservationCodec` ([`positronic/policy/codecs/observation.py`](observation.py)) is configured with state vectors to assemble (e.g. concatenate `robot_state.ee_pose` + `grip`) and images to resize. The same configuration builds the training columns and encodes the live observation, so the two match by construction.

**Action encoding** chooses what the model predicts and how that maps back to a robot command. This is where the action-space decisions live. Some real examples from [`positronic/policy/codecs/action.py`](action.py):

- **Absolute end-effector** (`AbsolutePositionAction`): the model predicts a target pose `[translation, rotation, grip]`. In training the label is the commanded EE pose in the chosen rotation representation; at inference `decode` turns the predicted vector into a `CartesianPosition` command. The model reasons in EE space, the robot is driven in EE space.
- **Absolute joints** (`AbsoluteJointsAction`): the model predicts joint angles `[q…, grip]` directly, and `decode` produces a `JointPosition` command — no inverse kinematics at runtime.
- **End-effector → joint targets via IK** (`IKJointsAction`): you recorded the robot in *end-effector* space (`robot_commands.pose`) but want a *joint-space* model. `IKJointsAction` runs inverse kinematics over each episode — seeded by the recorded `robot_state.q` — to compute the joint targets that reach those poses, and swaps them in as the action labels. You compose it with `|` ahead of `AbsoluteJointsAction`, which decodes the model's joint output at inference; `IKJointsAction` itself is then a pass-through, since the conversion only had to happen once, when building the training set. This is the sharpest illustration of the whole idea: the *same* end-effector recordings train either an EE-space or a joint-space model, just by changing the codec.

### Commanded vs observed targets

By default, action codecs label actions with the **commanded** targets (`robot_commands.pose`, `target_grip`) — "what the controller was told to do." The `_traj` variants instead use the **actual** robot trajectory (`robot_state.ee_pose`, `grip`) — "what the robot actually did" — and binarize the observed grip (continuous → open/close), since the model should learn a discrete grip. Same raw data, two different notions of the action label.

## Timing

Codecs return full action chunks without timestamps. The client processor
`ChunkedSchedule(fps, horizon_sec)` determines when commands run and how much of
each chunk executes. `training_fps` supplies training cadence metadata, independently of the deployment's playback `fps`.

## End-effector frames

"EE pose" has no universal meaning: our rigs report the frame their model calls `default`, DROID and RoboLab report the gripper frame (`droid_eef`). A checkpoint speaks whichever frame its training data was in, so serving it on a rig whose `default` sits elsewhere misreads every pose and every command by one constant transform — silently. See [the frame contract](../../drivers/roboarm/README.md) for what `default` is and what each embodiment owes it.

`ChangeEEFrame(T)` converts at that boundary: observations compose forward into the policy's frame (`pose * T`), commands compose back (`pose * T⁻¹`). `T` places the checkpoint's frame relative to `default`, so it belongs to the checkpoint and travels with it — nothing about the rig's model crosses the wire.

It is declared in two places, for the two things it does:

- **Training** — `compose(ee_frame=DROID_EE_FRAME)` re-expresses the dataset in that frame, which is what makes the resulting checkpoint speak it. It defaults to unset, which trains in `default`.
- **Serving** — the OpenPI pipeline's `ee_frame=` puts the codec left of the `remote` marker, so the rig converts and the server stays frame-agnostic. It has no default: every deployment states its frame — `None` for a checkpoint trained in `default`, or one that speaks joints, which are unambiguous. Nothing checks a stated frame against how the checkpoint was trained, so it is set beside the checkpoint path it belongs to.

Both take the transform itself — `models.DROID_EE_FRAME` is the one we ship — so a checkpoint declares its own frame and no robot model is consulted to serve it. GR00T's DROID codec uses `DROID_EE_FRAME` for both dataset conversion and serving. Its pipeline exposes this through `codec.ee_frame`.

A `CartesianDelta` is the one command this cannot convert on its own: a delta has no anchor pose, so it carries `frame` and the driver composes it where the measured pose lives.

## Control mode

`SetControlMode(mode)` stamps a control mode on every robot command of a decoded chunk, so a checkpoint executes under the law its training data ran under. It composes left of the action decoder: `SetControlMode(mode) | action`. `mode` is `Impedance(kq, kqd, kx, kxd)` or `PositionControl(stiffness=None)` from [`positronic.drivers.roboarm.command`](../../drivers/roboarm/command.py); a command without one runs under the arm's native law (see [the wire format](../../../docs/connect-your-model.md#actions-server--client)). Implementation in [`positronic/policy/codecs/action.py`](action.py).

Two wrappers in [`positronic/cfg/codecs.py`](../../cfg/codecs.py) apply it to an action codec:

| Wrapper | Expands to | Used by |
|---------|-----------|---------|
| `droid_execution(action)` | `SetControlMode(DROID_IMPEDANCE) \| action` ([the DROID gains](../../cfg/hardware/roboarm/__init__.py)) | the `droid` pipelines of OpenPI, DreamZero and MolmoAct2, and OpenPI's `droid_jointpos` |
| `phail_v1_execution(action)` | `SetControlMode(PositionControl()) \| action` | the `phail_v1` pipelines of LeRobot, OpenPI and DreamZero |

GR00T's DROID codec sets `DROID_IMPEDANCE` directly on its joint-position commands.

## Writing custom codecs

Subclass `positronic.policy.codecs.Codec` and implement `encode()` and/or `_decode_single()`. The base class returns `{}` from both — observation codecs override `encode()`, action codecs override `_decode_single()`. Middleware codecs that pass data through must explicitly `return data` (e.g. `BinarizeGripTraining`, a pure pass-through at decode that only binarizes via its `training_encoder`); middleware that transforms decoded actions modifies and returns `data` instead (e.g. `BinarizeGripInference`, which thresholds `target_grip` in `_decode_single`). Compose observation and action codecs with `&`, chain middleware with `|`. See the vendor codec files below for reference patterns.

## Codec catalog by vendor

The recipes below show how vendors configure and combine codecs. `compose` combines observation and action conversion, optional grip/frame conversion, and training cadence metadata. `ChunkedSchedule` owns inference cadence and the execution horizon; codecs return full chunks without timestamps.

At conversion time a codec is referenced by import path (`--dataset.codec=@positronic.vendors.<vendor>.codecs.<name>`). At serving time each vendor's server exposes its codecs as **named pipelines**, each one a server subcommand of the same name, so the same name selects the same codec on both sides.

### LeRobot (ACT — 0.3.3)

See [`positronic/vendors/lerobot_0_3_3/codecs.py`](../../vendors/lerobot_0_3_3/codecs.py).

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

See [`positronic/vendors/lerobot/codecs.py`](../../vendors/lerobot/codecs.py).

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

See [`positronic/vendors/gr00t/codecs.py`](../../vendors/gr00t/codecs.py).

| Codec | Cameras | State and training actions | Inference actions |
|-------|---------|----------------------------|-------------------|
| `droid` | Exterior + wrist | Absolute EEF pose (XYZ + row-based rot6d), gripper, 7 joints | Absolute joint targets + binary gripper |
| `droid_three_cameras` | Two exteriors + wrist | Same as `droid` | Same as `droid` |

Images use the upstream DROID client's 320×180 padded resize, then the checkpoint's native
preprocessing. GR00T owns pose-relative and joint-relative conversion. Training labels are
recorded state trajectories. Use the same camera layout for conversion and inference;
the published DROID checkpoint uses two cameras. See the [Docker workflow](../../vendors/gr00t/README.md).

### OpenPI

See [`positronic/vendors/openpi/codecs.py`](../../vendors/openpi/codecs.py).

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

Training and inference recipes must agree on state ordering, pose representation, image preparation,
gripper convention and action space. Their stacks can include different pieces: field renaming,
transport compression and scheduling apply at inference. Share the model conversion settings and
check these conventions when diagnosing a "Shape mismatch" or "Feature mismatch".

For the vendor pipelines below, select the same named recipe for conversion and serving:

```bash
# Training (ACT — 0.3.3)
cd docker && docker compose run --rm lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee

# Serve the matching recipe
cd docker && docker compose run --rm --service-ports lerobot-0_3_3-server ee
```

## See Also

- [Connect Your Model](../../../docs/connect-your-model.md) – the inference API and where codecs sit in it
- [Dataset Library README](../../dataset/README.md) – raw storage and transforms
- [Training Workflow](../../../docs/training-workflow.md) – using codecs in the pipeline
- Vendor docs: [LeRobot ACT](../../vendors/lerobot_0_3_3/README.md) | [SmolVLA](../../vendors/lerobot/README.md) | [GR00T](../../vendors/gr00t/README.md) | [OpenPI](../../vendors/openpi/README.md)
