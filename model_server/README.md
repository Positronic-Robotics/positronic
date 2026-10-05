# positronic-model-server

Serve a native model inside its own Python environment, without Positronic, Torch or JAX dependencies.
The wrapper owns transport, sessions, serialization and cleanup. Vendor scripts load and call their
model directly in the same process and describe the client pipeline as plain data.

```bash
uv pip install "positronic-model-server==0.2.0"
uv pip install "positronic-model-server[websocket]==0.2.0"
uv pip install "positronic-model-server[grpc]==0.2.0"
```

The core depends on NumPy, Pillow, msgpack and `positronic-wire`'s dependency-free transport
definitions. The extras select transport libraries independently; combine them as
`[websocket,grpc]` when both are needed. WebSocket serving uses Uvicorn directly; gRPC does not
require it. Starlette, AnyIO, FastAPI and Pydantic are not dependencies of either installation.
In a checkout, install the workspace packages together: `uv pip install ./wire ./model_server`.

## Serving and lifetime

This test server returns each observation as a one-element result list:

```python
from positronic_model_server import keys
from positronic_model_server.server import Model, ModelServer, Session
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.spec import component
from positronic_model_server.websocket_wire import WebsocketWire

def load_model():
    def prepare_session(params):
        if params["fps"] <= 0:
            raise ValueError("fps must be positive")
        return Session(
            infer=lambda observation: [observation],
            client_stack=component("chunked_schedule", version=2, fps=params["fps"]),
        )

    return Model(prepare_session, parameters={"fps": 20}, metadata={keys.CHECKPOINT_ID: "example"})

server = ModelServer(load_model, idle_timeout_min=10)
server.serve([WebsocketWire(ServedHostPort("0.0.0.0", 8000))])
```

Each vendor supplies its own script and argument parsing. For a native model, `load_model` loads
weights and finishes shared warm-up before returning. Listeners bind afterwards, so any successful
keepalive answer means shared startup has finished. The factory owns cleanup if loading raises.

`Model.parameters` declares accepted session names and defaults. Query values are decoded as JSON,
or retained as strings when they are not JSON. Duplicate or unknown names are refused; strings
never name Python imports. Every session gets an independent copy of the resolved parameters.
`prepare_session` decides how they affect inference and the client description, validates their
meaning and completes configuration-specific warm-up. It returns a `Session` only when the first
real inference needs no deferred loading or compilation. Warm-up must preserve the intended initial
session state. A preparation that raises owns cleanup of anything it acquired.

The server sends `waiting` messages during preparation, including while another session holds the
model. It then sends `ready` with a session ID, protocol v3, the pipeline description and metadata:

- `session_params`: the received query parameters, after decoding;
- `effective_params`: those parameters with declared defaults applied;
- `model_server_version`: this distribution's version;
- model/session metadata supplied by the vendor, plus the listener address.

The client may report any additional metadata and owns how it records or combines that information.
The server checks description structure without importing client components.

All model operations run on one worker thread in the server process: load, preparation, inference,
session cleanup and model cleanup. Each session's inference callable retains its own settings and
history. Inference is serialized across all listeners. Queue time is reported separately from model
time. The event loop answers keepalive while the worker is busy; no waiting messages interrupt an
inference response. A failed inference reports an error and leaves the session usable. A malformed
session request ends that session.

`Session.close` runs after outstanding work, on explicit close, disconnect or server shutdown.
The end-session acknowledgement follows cleanup. `Model.close` runs after sessions finish and
listeners stop, including when startup fails after loading. Call `server.shutdown()` from any thread;
an idle timeout ends the server only when no sessions remain. A model operation must return for
graceful shutdown to complete.

Pass `auth_token` to require a bearer token on both session and keepalive calls. `None` serves open;
empty or malformed tokens are refused at construction. For gRPC, use `grpc_wire.GrpcWire` with its
own `ServedHostPort`. Both listeners can be passed to one server and share its loaded model.

## Protocol compatibility

The shared server sends v3 results without interpreting robot-command fields or reshaping chunks.
Updated Positronic clients select this behavior while preserving v1/v2 command decoding for legacy
servers. A client that only supports v1/v2 rejects v3 during opening. Upgrade clients before migrating
their endpoint. Legacy servers and their server-side codecs remain supported in `positronic.offboard`.

## Client pipeline descriptions

Vendor code describes components as JSON-compatible data. Only the Positronic client imports
and instantiates their implementations:

```python
from positronic_model_server.spec import component, parallel, sequence

description = sequence(
    component("chunked_schedule", version=2, fps=20),
    parallel(
        component("restrict_image_size", width=320, height=180),
        component("metadata", values={"model": "example"}),
    ),
)
```

`component` accepts a name, explicit version and keyword arguments, normalizing arguments through
JSON. `sequence` and `parallel` reject empty compositions. Component names and argument meanings
belong to the client registry; these helpers do not resolve classes or validate model parameters.
The structural keys (`NAME`, `VERSION`, `ARGS`, `SEQ`, `PAR`) have one definition in `spec`.

## Serialization

```python
from positronic_model_server.serialization import deserialise, encode_jpeg, serialise

message = serialise({"camera": encode_jpeg(image, quality=90), "state": state})
restored = deserialise(message)
```

Plain values, nested mappings/sequences, NumPy arrays and NumPy scalars use the shared msgpack
representation. Ordinary arrays are lossless, including arrays whose dimensions resemble images.
JPEG compression is explicit: call `encode_jpeg` on the values to compress. It accepts RGB arrays
shaped `(..., H, W, 3)`, casts pixels to uint8 and preserves every leading dimension. It is lossy.
The receiver recognizes each encoded value without a field schema. The same functions encode
observations or model results, including returned images.

`serialization.JpegEncoding(path, quality)` selects a nested mapping/list value, with an empty path
selecting the whole result. Supply these selectors through `Session.output_images` for returned
images. `serialization.encode_images` applies the same selections to observations without changing
the source containers. A missing path fails instead of silently skipping compression. The client
component `encode_images` accepts `paths` and `quality`; its result decoding leaves native values
unchanged. V3 does not infer image fields from array dimensions.

Nonempty 3D and 4D images retain the v1/v2 JPEG representation. Extra leading dimensions and empty
batches carry a full-shape marker; send those only to receivers supporting it. NumPy object,
structured and complex dtypes are unsupported. Python tuples arrive as lists.

`protocol` owns shared message fields, authentication, timing and protocol version names.
Serialization here never interprets robot-command fields or `__cmd__` envelopes. Positronic's
legacy offboard protocol owns those hooks and its supported-version selection.

## Package checks

Run `uv run pytest model_server positronic/offboard/tests positronic/policy/tests wire` from the
repository root. CI also builds wheels and installs them in isolated environments with NumPy
1.26.4 on Python 3.11 and 3.12, selecting no transport, WebSocket alone, or gRPC alone. Root tests
exercise the repository's locked NumPy version.
