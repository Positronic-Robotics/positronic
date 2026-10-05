# positronic-model-server

Shared inference protocol definitions, pipeline descriptions and value serialization for model
environments. Install it without Positronic, Torch or JAX. The package contains data utilities;
it does not provide a server entrypoint or session lifecycle.

```bash
uv pip install "positronic-model-server==0.1.1"
uv pip install "positronic-model-server[websocket]==0.1.0"
uv pip install "positronic-model-server[grpc]==0.1.0"
```

The core depends on NumPy, Pillow, msgpack and `positronic-wire`'s dependency-free transport
definitions. The extras select transport libraries independently; combine them as
`[websocket,grpc]` when both are needed. HTTP server dependencies are not part of these extras.
In a checkout, install the workspace packages together: `uv pip install ./wire ./model_server`.

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
