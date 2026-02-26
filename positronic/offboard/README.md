# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Protocol v1

The unified WebSocket protocol is built to enable ANY hardware to connect to ANY model. All Positronic inference servers (LeRobot, GR00T, OpenPI) implement this protocol, allowing a single `.remote` policy client to work across all vendors.

### Endpoints

#### `GET /api/v1/models`
Returns a list of available model IDs.

**Example Request:**
```bash
curl http://localhost:8000/api/v1/models
```

**Response:**
```json
{
  "models": ["10000", "20000", "30000"]
}
```

Use this to discover which models are available before connecting.

#### `WS /api/v1/session`
Establishes an inference session with the **default** model (latest checkpoint for currently available vendors).

#### `WS /api/v1/session/{model_id}`
Establishes an inference session with a **specific** model.

**Example:**
- `ws://localhost:8000/api/v1/session` → Latest model
- `ws://localhost:8000/api/v1/session/10000` → Model 10000
- `ws://localhost:8000/api/v1/session/20000` → Model 20000

### WebSocket Flow

#### 1. Handshake
Upon connection, the server sends a metadata packet:

```json
{
  "meta": {
    "type": "lerobot",
    "host": "localhost",
    "port": "8000",
    "checkpoint_path": "~/checkpoints/lerobot/experiment_v1",
    "checkpoint_id": "10000",
    "image_sizes": [224, 224],
    "action_fps": 15.0,
    "action_horizon_sec": 1.0
  }
}
```

This metadata tells the client:
- Which checkpoint is loaded
- Server connection details
- Codec metadata (`image_sizes` for client-side resize, `action_fps` and `action_horizon_sec` for timing)

#### 2. Status Updates (Long Model Loading)

Some models may take a long time to load (e.g., OpenPI and GR00T can take 120-300s). The server sends periodic status updates during loading to prevent WebSocket keepalive timeouts:

```json
{
  "status": "loading",
  "message": "Loading checkpoint 10000, please wait..."
}
```

The client should display these status updates to the user. Once loading completes, the server sends the metadata packet.

#### 3. Inference Loop

After handshake, the client streams observations and receives actions:

**Client → Server (Observation):**
```json
{
  "ee_pose": [0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
  "grip": [0.04],
  "wrist_image": "<base64_encoded_image>",
  "exterior_image": "<base64_encoded_image>"
}
```

**Server → Client (Action):**
```json
{
  "result": [{
    "action": {
      "target_pose": [0.51, 0.21, 0.31, 0.0, 0.0, 0.0, 1.0],
      "target_grip": [0.02]
    }
  }]
}
```

**Server → Client (Error):**
```json
{
  "error": "Shape mismatch: expected (7,) but got (6,)"
}
```

The loop continues until the client closes the connection or the episode ends.

### Key Benefits

**Unified API:** All vendors implement the same protocol, so swapping models is as simple as changing the server:

```bash
# LeRobot server
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/exp_v1 \
  --codec=@positronic.vendors.lerobot.codecs.ee

# GR00T server (swap hardware code stays the same)
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/exp_v1

# Client connects the same way
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost
```

**Model Switching:** Compare multiple models without restarting the server by using specific session endpoints.

**Status Streaming:** Long model loads are handled gracefully with progress updates.

**Python Client:** We provide a Python client (`positronic.offboard.client.InferenceClient`) that handles the WebSocket protocol automatically. While the API is currently in alpha and may change, we'll do our best to maintain backward compatibility for the inference client.

## Classes

### `basic_server.InferenceServer`
A generic server that serves policies from a provided registry.

```python
from positronic.offboard.basic_server import InferenceServer

registry = {
    'model_a': lambda: load_model_a(),
    'model_b': lambda: load_model_b()
}

server = InferenceServer(registry, host='0.0.0.0', port=8000)
# Default session (/api/v1/session) connects to the first model in registry
server.serve()
```

### `client.InferenceClient`
A Python client for connecting to an inference server.

```python
from positronic.offboard.client import InferenceClient

client = InferenceClient('localhost', 8000)

# Connect to default policy
session = client.new_session()
# OR connect to specific policy
# session = client.new_session('model_a')

meta = session.metadata
action = session.infer(observation)
```

## Vendor Implementations

All vendor servers implement Protocol v1:

- **LeRobot**: `positronic.vendors.lerobot.server` - Serves ACT checkpoints with dynamic loading
- **GR00T**: `positronic.vendors.gr00t.server` - Serves GR00T checkpoints with modality config
- **OpenPI**: `positronic.vendors.openpi.server` - Serves OpenPI checkpoints with config name

Each server enforces a **Singleton Policy** (only one checkpoint loaded at a time) to manage GPU resources efficiently.

## See Also

- [Training Workflow](../../docs/training-workflow.md) - Starting inference servers
- [Inference Guide](../../docs/inference.md) - Remote policy usage and patterns
- [Model Selection](../../docs/model-selection.md) - Choosing between vendors
