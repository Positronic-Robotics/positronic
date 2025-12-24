# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Protocol v1

The protocol supports discovering available models and establishing specific inference sessions.

### Endpoints

#### `GET /api/v1/models`
Returns a list of available model IDs.

**Response:**
```json
{
  "models": ["act/100", "act/200"]
}
```

#### `WS /api/v1/session`
Establishes an inference session with the **default** policy.

#### `WS /api/v1/session/{model_id}`
Establishes an inference session with a **specific** policy.

### WebSocket Flow
1.  **Handshake**: Upon connection, the server sends a metadata packet.
    ```json
    {
      "meta": {
        "host": "...",
        "port": "...",
        "checkpoint_path": "..."
      }
    }
    ```
2.  **Inference**: The client streams observations, and the server replies with actions.
    *   **Client -> Server**: Serialized observation dict.
    *   **Server -> Client**: `{"result": [{"action": ...}]}` or `{"error": "..."}`.

---

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

-   **Lerobot**: `positronic.vendors.lerobot.server` implements this protocol for dynamically serving checkpoints from a directory, enforcing a **Singleton Policy** (only one loaded at a time) to manage GPU resources.
