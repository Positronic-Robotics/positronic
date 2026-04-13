# Connect Your Model

Run your policy against Positronic's simulation environment, or connect it for real-robot evaluation on [PhAIL](https://phail.ai).

This guide covers:

1. Running a reference model to see the system end-to-end
2. Understanding observations, actions, and codecs
3. Implementing your own inference server

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/)
- Clone the repo: `git clone git@github.com:Positronic-Robotics/positronic.git && cd positronic`

## Run a Reference Model

Start an ACT inference server using a public checkpoint trained on the simulated cube stacking task:

```bash
cd docker && docker compose run --rm --service-ports lerobot-0_3_3-server demo
```

The server downloads the checkpoint (~505MB) and starts a WebSocket API on port 8000. The server requires Docker on Linux. Verify it's ready:

```bash
curl http://localhost:8000/api/v1/models
# {"models": ["050000"]}
```

In a separate terminal, run inference in MuJoCo simulation. The inference client runs on Mac or Linux:

```bash
uv run positronic-inference sim \
  --policy=.remote --policy.host=<server-host> --policy.port=8000 \
  --driver.show_gui=True \
  --output_dir=~/datasets/demo_run
```

The MuJoCo window shows the Franka arm executing the policy in real time. The `--output_dir` flag records all episodes (robot state, camera feeds, actions) for later review.

Browse recorded episodes with:

```bash
uv run positronic-server --dataset.path=~/datasets/demo_run --port=5001
# Open http://localhost:5001
```

## Observations and Actions

Every timestep, the inference client sends the current robot state to the server and receives a chunk of actions back. All messages use [msgpack](https://msgpack.org/) with numpy array support (see [Serialization](#serialization) below).

### Observations (client to server)

The client sends the full raw robot state as a dict:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `ee_pose` | float32 | (7,) | End-effector pose: x, y, z, qx, qy, qz, qw |
| `joints` | float32 | (7,) | Joint positions (radians) |
| `grip` | float32 | (1,) | Gripper opening |
| `wrist_image` | uint8 | (H, W, 3) | Wrist camera RGB |
| `exterior_image` | uint8 | (H, W, 3) | External camera RGB |

Your server receives all fields. Use what your model needs, ignore the rest.

### Actions (server to client)

The server returns a list of action dicts (an action chunk). The client executes them sequentially at the configured action frequency:

```python
{"result": [{"robot_command": {...}, "target_grip": 0.04}, ...]}
```

The `robot_command` field specifies the control mode:

| Command type | Fields | Description |
|--------------|--------|-------------|
| `cartesian_pos` | `pose`: float32 (12,) | Target EE pose (position + rotation matrix) |
| `joint_pos` | `positions`: float32 (7,) | Target joint angles (radians) |
| `joint_delta` | `velocities`: float32 (7,) | Joint velocity command |

The codec determines which command type the model produces (see below).

## Codecs: State and Action Representations

Different models expect different input/output formats. Some use end-effector pose, others use joint positions. Some output absolute targets, others output deltas. Positronic uses **codecs** to handle this translation.

A codec sits between the wire protocol and the model:

```
Raw observation (wire) --> codec.encode() --> model input
Model output           --> codec.decode() --> raw action (wire)
```

The wire format (what your server receives and returns) is always the raw robot state described above. If you use Positronic's built-in servers, the codec is configured at server startup:

```bash
# EE pose observation, absolute position actions
docker compose run --rm --service-ports lerobot-0_3_3-server serve \
  --checkpoints_dir=... \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee

# Joint position observation
docker compose run --rm --service-ports lerobot-0_3_3-server serve \
  --checkpoints_dir=... \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.joints
```

If you implement your own server, you handle this transformation yourself: pick the fields you need from the raw observation, and return actions in the raw format.

### Common representations

| Observation space | What the model sees | When to use |
|-------------------|--------------------|----|
| EE pose (7D) + grip + images | Position and orientation of the end-effector | Most common; sufficient for most manipulation tasks |
| EE pose + joint positions (7D) + grip + images | Both EE and joint state | When joint configuration matters (redundancy resolution, singularity avoidance) |
| Joint positions (7D) + grip + images | Joint angles only | Joint-space policies; no EE computation needed |

| Action space | What the model outputs | When to use |
|--------------|----------------------|-----|
| Absolute EE position (7D) + grip | Target pose the robot should move to | Default; works with position controllers |
| EE delta + grip | Displacement from current pose | Relative policies; smaller action space |
| Joint positions (7D) + grip | Target joint angles | Direct joint control; bypasses IK |

All built-in codecs are documented in the [Codecs Guide](codecs.md) with vendor-specific variants.

## Implement Your Own Server

To connect a custom model, implement a WebSocket server that speaks Positronic's Protocol v1.

### Endpoints

Your server must expose:

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /api/v1/models` | HTTP | Returns `{"models": ["model_a", "model_b"]}` |
| `WS /api/v1/session` | WebSocket | Inference session with default model |
| `WS /api/v1/session/{model_id}` | WebSocket | Inference session with specific model |

### Session Flow

1. Client connects via WebSocket
2. Server sends status messages while loading (optional but recommended for slow loads):
   ```python
   {"status": "loading", "message": "Loading model..."}
   ```
3. Server sends ready with metadata:
   ```python
   {"status": "ready", "meta": {"type": "my_model", "checkpoint_id": "v1"}}
   ```
4. Inference loop: client sends observation, server returns action, repeat until disconnect

### Using Positronic's Server Base Class

The simplest way is to subclass `InferenceServer` and provide a policy:

```python
from positronic.offboard.basic_server import InferenceServer
from positronic.policy import Policy

class MyPolicy(Policy):
    def __init__(self, model):
        self._model = model

    def select_action(self, obs):
        # obs contains: ee_pose, joints, grip, wrist_image, exterior_image
        # Pick what your model needs:
        images = obs['exterior_image']
        ee = obs['ee_pose']

        # Run your model, get a list of predicted poses
        predicted_poses = self._model.predict(images, ee)

        # Return action chunk: list of commands
        return [
            {'robot_command': {'type': 'cartesian_pos', 'pose': pose}, 'target_grip': 0.04}
            for pose in predicted_poses
        ]

    def reset(self, context=None):
        pass

    @property
    def meta(self):
        return {'type': 'my_model'}

# Create server with policy registry
server = InferenceServer(
    policy_registry={'default': lambda: MyPolicy(load_my_model())},
    host='0.0.0.0',
    port=8000,
)
server.serve()
```

Test it:

```bash
uv run positronic-inference sim \
  --policy=.remote --policy.host=localhost --policy.port=8000
```

### Standalone Implementation

If you prefer not to depend on Positronic for the server, implement the WebSocket protocol directly. The key requirement is msgpack serialization with numpy support (see below).

### Serialization

All messages use msgpack. Numpy arrays are encoded with a custom extension:

```python
# numpy array -> msgpack
{
    b"__ndarray__": True,
    b"data": array.tobytes(),   # raw bytes
    b"dtype": str(array.dtype), # e.g. "<f4"
    b"shape": array.shape       # tuple
}
```

Positronic provides `serialise()` and `deserialise()` in `positronic.utils.serialization` that handle this automatically:

```python
from positronic.utils.serialization import serialise, deserialise

# Server-side WebSocket handler
async for message in websocket.iter_bytes():
    obs = deserialise(message)           # dict with numpy arrays
    action = policy.select_action(obs)
    await websocket.send_bytes(serialise({"result": action}))
```

## See Also

- [Inference Guide](inference.md) – local and remote inference patterns
- [Codecs Guide](codecs.md) – all available codecs by vendor
- [Offboard Protocol](../positronic/offboard/README.md) – full Protocol v1 specification
- [Training Workflow](training-workflow.md) – training with public datasets
