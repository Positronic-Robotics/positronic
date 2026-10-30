# Direct API robot policy

`LLMPolicy` runs on the rig and calls a public model API on a worker thread. The same policy can also run behind an inference server; see [Serve as an endpoint](#serve-as-an-endpoint). Pydantic AI selects the provider and manages its client, native messages, and SDK defaults; the policy owns the conversation, tools, and validation. The runner owns episode boundaries.

## Run in simulation

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the rig's environment. Choose a model that accepts images and function calls.

```bash
uv sync --extra llm-openai
uv run --extra llm-openai positronic eval run \
  --eval=.sim.positronic.stack_cubes --eval.timeout=300 \
  --policy=@positronic.vendors.llm.policy.llm \
  --policy.model="openai-responses:$OPENAI_MODEL" \
  --policy.settings="{'openai_store': False}" \
  --output_dir=~/datasets/llm-policy/sim
```

Use `--charge_inference_time=True` to count API latency against simulated trial time, or `--charge_inference_time=False` to pause simulated time while waiting for inference. Hardware always pays wall time. Use separate output directories when comparing the two modes.

### Models and dependencies

`--policy.model` takes Pydantic AI's `provider:model` identifier. Pydantic AI loads the selected provider and reads its standard credential environment variables. MysteryBox can populate those variables before starting the rig.

| Model prefix | Install extra | Credential environment variable |
| --- | --- | --- |
| `openai-responses:` or `openai-chat:` | `llm-openai` | `OPENAI_API_KEY` |
| `anthropic:` | `llm-anthropic` | `ANTHROPIC_API_KEY`, or a pool in `ANTHROPIC_API_KEYS` |
| `google:` | `llm-google` | `GOOGLE_API_KEY` (also accepts `GEMINI_API_KEY`) |

The `llm` extra installs the Pydantic AI core and Pillow for image encoding. Each provider extra adds only that provider's SDK dependencies. Other [Pydantic AI providers](https://ai.pydantic.dev/models/overview/) work without policy changes when their dependencies are installed. Choose a model that supports images and function calls; incompatible models raise errors.

To spread `anthropic:` episodes over the keys of different organizations, set `ANTHROPIC_API_KEYS` to a comma-separated list of keys. Each episode uses one key from the list at random; a model name with no prefix that pydantic-ai reads as Anthropic, such as `claude-sonnet-4-5`, uses the list too. When the variable is set, `ANTHROPIC_API_KEY` has no effect on `anthropic:` models. A list that is empty or has a blank entry stops the episode with an error. The pool does not apply to a `gateway/anthropic:` model, which uses the gateway key, or to a configured `Model`, which uses its own provider. Do not put keys in arguments or config values: the episode records those.

Use provider environment variables for supported endpoint overrides, such as `OPENAI_BASE_URL` for an OpenAI-compatible service. For further customization, a Python config can pass an already configured Pydantic AI `Model` to `llm(model=...)`. For example:

```python
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from positronic.vendors.llm.policy import llm

policy = llm(
    model=OpenAIChatModel(
        os.environ['LLM_MODEL'],
        provider=OpenAIProvider(
            base_url='https://your-provider.example/v1',
            api_key=os.environ['YOUR_KEY_VARIABLE'],
        ),
    ),
)
```

`--policy.settings` accepts [Pydantic AI model settings](https://ai.pydantic.dev/api/settings/), including provider-specific options. These override defaults on a configured model and are recorded with the episode. Put authentication and transport configuration on the provider, not in these recorded settings. Endpoint URLs must use HTTP(S) and have no userinfo, query parameters, or fragments.

Provider-specific behavior follows the library defaults. The example explicitly sets `openai_store=False` to disable OpenAI response storage. Conversation history belongs to each episode; do not configure server-side conversation IDs shared across episodes.

## Control contract

The default config runs `LLMPolicy`. Each episode runs a processor generator that yields `Step` values with commands and the next requested call time. A decision sees one frozen observation. The control thread prepares the messages and images, then submits one model invocation to the worker runtime. SDK network retries belong to that invocation. The processor consumes the reply on a later control-loop call. It can request a correction or picture, play a move, or end the episode. Picture and correction requests share the frozen observation. Once the waypoints play, the next decision receives measured state. Each observation includes the previous target and remaining translation/gripper error.

Only the task instruction, measured hand pose/gripper, and selected labelled RGB images enter the model prompt. Privileged simulator state and ground-truth success do not. Positions use the measured pose's coordinate frame, in metres. Orientations use roll/pitch/yaw in radians, with `R = Rz(yaw) Ry(pitch) Rx(roll)`. Gripper values run from 0 (open) to 1 (closed).

| Tool | Arguments / effect |
| --- | --- |
| `move_to` | Absolute `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper`, and a short `note` |
| `done` | `reason` and `hindsight`; end the episode with `PolicyEnd.COMPLETE` |
| `give_up` | `reason` and `hindsight`; end the episode with `PolicyEnd.GIVE_UP` |
| `reveal_frames` | `cameras` and `note`; reveal selected frames in `images=on_demand` mode |

The model must return exactly one tool call. Malformed arguments, unavailable tools, and multiple calls receive explicit correction feedback. Oversized moves are clamped and accepted with the bounded target in the tool result. Three consecutive invalid replies raise an error. SDK retry defaults apply, and terminal API errors propagate. The overall timeout covers the invocation and its retry waits. The episode has a budget of 100 model invocations, including pictures and corrections; individual network attempts within an invocation do not consume additional budget. Exhaustion ends the episode with `PolicyEnd.GIVE_UP`.

After `done`, `give_up`, or call-budget exhaustion, the run returns its `PolicyEnd`, and the harness ends the episode at once. The episode records `eval.ended_by = policy` and `eval.policy_end`. The `PolicyEnd` is the claim of the model, not a verdict. Each new episode gets a fresh conversation.

The policy's stop reason and hindsight are recorded for inspection, without setting a success label or carrying advice into other episodes. There are no code-execution, crop, VLA, or skill-selection tools.

### Motion and images

Defaults are experimental bounds for supervised testing:

| Setting | Default |
| --- | --- |
| `motion.max_translation` | 0.05 m per move |
| `motion.max_rotation` | 0.3491 rad (20°) per move |
| `motion.linear_speed` | 0.05 m/s |
| `motion.angular_speed` | 0.5236 rad/s (30°/s) |
| `motion.fps` | 25 waypoints/s |
| `timeout` | 120 seconds per model invocation, including retries |
| `max_calls` / `max_invalid` | 100 / 3 |
| `camera_keys` | `image.wrist`, `image.exterior` |
| `image_size` | 640 pixels on the longest edge; aspect ratio preserved |
| `image_horizon` | Frames of the most recent 3 observations |
| `images` | `always` |

Set, for example, `--policy.motion.max_translation=0.02` or `--policy.images=on_demand`. On-demand pictures reveal frames from the current frozen observation; they do not advance time or move a camera. A camera cannot be revealed twice during one decision. The history keeps the frames of the last `image_horizon` observations. When an observation leaves that window, its frames change to the text `[older camera frame omitted]`, once. Each request carries a cache point after the newest changed frame message and one at its end. A request that ends in tool results has its end point on the newest user prompt instead. Pydantic AI sends a cache point to Anthropic as `cache_control`. With `--policy.cache_breakpoints=True`, an `openai-responses:` model gets each cache point as `prompt_cache_breakpoint` on the block before it. Only the GPT-5.6 family and later accept it: an earlier model answers with HTTP 400, so the flag is off by default. Without the breakpoint, Azure OpenAI reads no cache after a frame changes. A configured `OpenAIResponsesModel` drops cache points, so with the flag the policy refuses it. Configure a `BreakpointResponsesModel` from `positronic.vendors.llm.responses` instead. Azure OpenAI refuses a request with more than 50 images.

Translation is linear and rotation follows the shortest spherical interpolation. A returned move starts from the latest measured pose. Oversized translation and rotation are clamped independently to the per-move limits, preserving the translation direction and shortest rotation path. The processor waits one sampling period after the final command before requesting the next decision. The tool result and recorded acceptance contain the bounded target, whether it was clamped, and the trajectory duration including that final period. The next observation reports that target alongside the measured state; scheduling a target does not confirm arrival. Malformed tool arguments still require correction. The driver performs inverse kinematics and enforces its own joint constraints. These bounds are not collision detection or contact-force limits.

## Recordings and lifecycle

Each episode buffers a compact transcript. With `--output_dir`, the episode recorder saves the event list as `inference.policy.transcript` in the episode's `static.json` when the episode finishes. Model configuration, stop reason, and hindsight are stored alongside it in the policy metadata. An episode that used a key from `ANTHROPIC_API_KEYS` records `inference.policy.api_key_fingerprint`: the first 8 hex digits of the SHA-256 of that key. To split cost by organization, compare it with the fingerprint of each key: `printf %s "$KEY" | sha256sum | cut -c1-8`. The key itself is not recorded. Endpoint URLs are not recorded. Without an output directory, the transcript remains in memory.

Events contain the system prompt and tool schemas, measured observations, call numbers, tool replies and text, SDK token usage, rejections, and accepted decisions. Camera names and observation timestamps refer to the recorded image signals. Images, repeated conversation history, raw HTTP bodies, and provider reasoning signatures are excluded from the transcript. The model's live conversation retains the images and native reasoning metadata needed for subsequent API calls.

The processor records responses when it consumes them. The recorded snapshot contains events available when the episode finishes; an unconsumed response is absent even if its API call has completed. Later responses and generator cleanup do not modify it. Failed or aborted episodes need not retain a transcript.

Only one model invocation can be in flight. The processor continues consuming replies and playing commands while the robot reports an unavailable status. Playback combines overdue waypoints into the latest command. API failures propagate. Follow-up invocations start only when the control loop resumes the processor. An ended episode starts no further invocations, and a late response cannot command motion. The runtime waits for the current invocation, including SDK retries within its timeout, before closing the generator.

## Serve as an endpoint

`positronic.vendors.llm.server` serves the policy over the [offboard protocol](../../offboard/README.md). One process serves one model, and the API key comes from the server's environment. Each session opens its own model client at its first call, and closes it when the session ends or fails. So with `ANTHROPIC_API_KEYS` set, each session uses one key from the list at random, as each local episode does. `--policy.*` takes the same flags as the local config:

```bash
uv run --locked --extra llm-anthropic python -m positronic.vendors.llm.server \
  --policy.model=anthropic:claude-opus-5-5 --policy.motion.max_translation=0.02
```

The server listens on port 8000 and takes the wire flags of `positronic.offboard.server.serve`. A client connects as to any other server:

```bash
uv run --locked positronic eval run --eval=.sim.positronic.stack_cubes --eval.timeout=300 \
  --policy=.remote --policy.address.host=<server> --policy.address.port=8000 \
  --policy.infer_timeout=600 --output_dir=~/datasets/llm-policy/remote
```

Each session has its own conversation, and the end of the session drops it. One inference call is one decision, and it includes all corrections and pictures. A decision makes at most `max_invalid` API calls with `images=always`, and `(cameras + 1) × max_invalid` with `images=on_demand`, each up to `timeout` seconds. `--decision_timeout` (360 s by default) stops a decision before its next call, so one decision takes at most `decision_timeout + timeout` seconds: 480 s by default. Set the client's `--policy.infer_timeout` above that: 600 s here. The server sees a dropped client only when it sends the answer, so a reconnect during a decision waits for the dropped decision first, up to twice that bound. A decision that raises or passes its deadline fails the session, and every later call of that session fails. A move answers with its trajectory. The client plays it at `motion.fps` with the local cadence, and then asks for the next decision. A move starts from the pose in the request, not from the pose when the answer arrives. `done`, `give_up` and call-budget exhaustion answer with the `end` field, and the episode records `eval.policy_end`.

The server runs one call at a time, so concurrent episodes wait for each other. Declare a platform endpoint for it with `max_concurrent_sessions: 1`, and start one server process for each parallel episode. The server runs no warmup inference, because each call to the provider is paid.

### Find the transcript of a remote episode

The episode records the model configuration under `inference.policy.server`, and the session ID under `inference.policy.session_id`. The server writes the rest of what a local episode records to `<sessions_dir>/<session_id>.json` on the server host: the policy meta, `stop_reason`, `hindsight`, `transcript` with the token usage of each response, `session_id`, and `api_key_fingerprint` when the session used a key from `ANTHROPIC_API_KEYS`. `--sessions_dir` is `llm_sessions` under the server's working directory by default, and the server logs the absolute path when it starts. It writes the file when the session decides its end or fails, and again when the session closes. So a session that the client ends by a timeout or a disconnect also has a file, without `stop_reason`. A connection that drops before its first answer reconnects under a new session ID, and the dropped session keeps a file of its own. The episode records the new ID once the new session has answered; an episode that ends first, or whose new session fails, records the dropped ID. The server stamps each observation with its own monotonic clock, so the `obs_time_ns` values in a remote transcript do not match the times of the recorded signals.

## Supervised hardware

After validating the task in simulation, use the same policy with `positronic-inference real` and the DROID embodiment. Set the task instruction and output directory explicitly. The keyboard operator controls episode start/stop; the hardware's independent emergency stop remains available. This policy supports the single-arm hand-pose contract, without camera calibration or a workspace/collision model.

## Tests

```bash
uv run --extra llm pytest positronic/vendors/llm/tests
uv run --extra llm-openai --extra llm-anthropic pytest positronic/vendors/llm/tests
```

The core suite uses Pydantic AI's in-process test model without provider SDKs. Mocked OpenAI and Anthropic HTTP tests check endpoint configuration, images, tools, cache points, and client cleanup; each skips when its provider SDK is absent. Policy tests cover history retention, deadlines, errors, motion, budgets, and recording, including the full harness with a deterministic robot. Provider-specific serialization and SDK retries are handled by Pydantic AI. Tests need no API credentials or hardware.
