# Direct API robot policy

`LLMPolicy` runs on the rig and calls a public model API on a worker thread. There is no Positronic inference server. Pydantic AI selects the provider and manages its client, native messages, and SDK defaults; the policy owns the conversation, tools, and validation. The runner owns episode boundaries.

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
| `anthropic:` | `llm-anthropic` | `ANTHROPIC_API_KEY` |
| `google:` | `llm-google` | `GOOGLE_API_KEY` (also accepts `GEMINI_API_KEY`) |

The `llm` extra installs only the Pydantic AI core. Each provider extra adds only that provider's SDK dependencies. Other [Pydantic AI providers](https://ai.pydantic.dev/models/overview/) work without policy changes when their dependencies are installed. Choose a model that supports images and function calls; incompatible models raise errors.

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

`--policy.settings` accepts [Pydantic AI model settings](https://ai.pydantic.dev/api/settings/), including provider-specific options. These override defaults on a configured model and are recorded with the episode. Put authentication and transport configuration on the provider, not in these recorded settings. Model URLs containing credentials or query parameters are rejected.

Provider-specific behavior follows the library defaults. The example explicitly sets `openai_store=False` to disable OpenAI response storage. Conversation history belongs to the session; do not configure server-side conversation IDs shared across episodes.

## Control contract

The default config wraps the policy in `StopOnFault | ChunkedSchedule`. A decision sees one frozen observation. Each worker job encodes any requested images and makes one model invocation, which may include SDK network retries. The session processes the reply on a later control-loop call and decides whether to request a correction or picture, play a move, or remain idle. Picture and correction requests share the decision's frozen observation. Once the scheduled waypoints play, the next decision receives measured state again. An accepted target does not prove that the hand arrived: each subsequent observation includes the previous target and remaining translation/gripper error.

Only the task instruction, measured hand pose/gripper, and selected labelled RGB images enter the model prompt. Privileged simulator state and ground-truth success do not. Positions use the measured pose's coordinate frame, in metres. Orientations use roll/pitch/yaw in radians, with `R = Rz(yaw) Ry(pitch) Rx(roll)`. Gripper values run from 0 (open) to 1 (closed).

| Tool | Arguments / effect |
| --- | --- |
| `move_to` | Absolute `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper`, and a short `note` |
| `done` | `reason` and `hindsight`; stop issuing actions for this episode |
| `give_up` | `reason` and `hindsight`; stop issuing actions for this episode |
| `take_pic` | `cameras` and `note`; reveal selected frames in `images=on_demand` mode |

The model must return exactly one tool call. Malformed arguments, unavailable tools, and multiple calls receive explicit correction feedback. Oversized moves are clamped and accepted with the bounded target in the tool result. Three consecutive invalid replies raise an error. SDK retry defaults apply, and terminal API errors propagate. The overall timeout covers the invocation and its retry waits. The episode has a budget of 100 model invocations, including pictures and corrections; individual network attempts within an invocation do not consume additional budget. Exhaustion stops further actions and model invocations.

After `done`, `give_up`, or call-budget exhaustion, the session returns an empty trajectory on every call and makes no further API requests. Queued commands are cleared; drivers retain their last commanded target. The episode and recording continue until the simulator or operator ends it, or its timeout expires. An episode without a timeout requires external completion. Cancellation does not restart a finished session; each new episode gets a fresh session.

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
| `image_horizon` | Most recent 2 observations containing images |
| `images` | `always` |

Set, for example, `--policy.motion.max_translation=0.02` or `--policy.images=on_demand`. On-demand pictures reveal frames from the current frozen observation; they do not advance time or move a camera. A camera cannot be revealed twice during one decision. Older images are removed from outgoing history while text, tool results, and native reasoning metadata remain.

Translation is linear and rotation follows the shortest spherical interpolation. A returned move starts from the latest measured pose. Oversized translation and rotation are clamped independently to the per-move limits, preserving the translation direction and shortest rotation path. The tool result and recorded acceptance contain the bounded target, whether it was clamped, and the trajectory duration. The next observation reports that target alongside the measured state; scheduling a target does not confirm arrival. Malformed tool arguments still require correction. The driver performs inverse kinematics and enforces its own joint constraints. These bounds are not collision detection or contact-force limits.

## Recordings and cancellation

Each session buffers a compact transcript and exposes a snapshot through `Session.meta`. With `--output_dir`, the episode recorder saves the event list as `inference.policy.transcript` in the episode's `static.json` when the episode finishes. Model configuration, stop reason, and hindsight are stored alongside it in the policy metadata. Without an output directory, the transcript remains in memory.

Events contain the system prompt and tool schemas, measured observations, call numbers, tool replies and text, SDK token usage, rejections, and accepted/discarded decisions. Camera names and observation timestamps refer to the recorded image signals. Images, repeated conversation history, raw HTTP bodies, and provider reasoning signatures are excluded from the transcript. The model's live conversation retains the images and native reasoning metadata needed for subsequent API calls.

The recorded snapshot contains events available when the episode finishes. Later responses and session cleanup do not modify it. Failed or aborted episodes need not retain a transcript.

Only one model invocation can be in flight. Faults mark its reply for discard. Follow-up invocations start only when the control loop calls the session. An ended episode starts no further invocations, and a late response cannot command motion. The runtime waits for the current invocation, including any SDK retries within its timeout, before the session closes. Cancellation does not promise to stop provider billing for a request already sent.

## Supervised hardware

After validating the task in simulation, use the same policy with `positronic-inference real` and the DROID embodiment. Set the task instruction and output directory explicitly. The keyboard operator controls episode start/stop; the hardware's independent emergency stop remains available. This policy supports the single-arm hand-pose contract, without camera calibration or a workspace/collision model.

## Tests

```bash
uv run --extra llm pytest positronic/vendors/llm/tests
uv run --extra llm-openai --extra llm-anthropic --extra llm-google pytest positronic/vendors/llm/tests
```

The core suite runs without provider SDKs. Provider integration tests skip when their SDK is absent; installing all three extras exercises every HTTP adapter. Tests mock HTTP, check native reasoning/image replay and client cleanup, validate retries, deadlines, motion and cancellation, and run the full harness/recorder with a deterministic robot. They need no API credentials or hardware.
