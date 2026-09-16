# Direct API robot policy

`LLMPolicy` runs on the rig and calls a public model API on a worker thread. There is no Positronic inference server. Pydantic AI supplies native request and message adapters; the policy owns the conversation, tools, and validation. The runner owns episode boundaries.

## Run in simulation

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the rig's environment. Choose a model that accepts images and function calls.

```bash
uv sync --extra llm
uv run --extra llm positronic eval run \
  --eval=.sim.positronic.stack_cubes --eval.timeout=300 \
  --policy=@positronic.vendors.llm.policy.llm \
  --policy.model="$OPENAI_MODEL" \
  --output_dir=~/datasets/llm-policy/sim
```

Add `--charge_inference_time=True` to count API latency against simulated trial time. The default pauses simulated time while waiting for inference. Hardware always pays wall time. Use separate output directories when comparing the two modes.

| `--policy.api` | Native API | Default key environment variable |
| --- | --- | --- |
| `openai-responses` (default) | OpenAI Responses | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages | `ANTHROPIC_API_KEY` |
| `google` | Google generateContent | `GEMINI_API_KEY` |
| `openai-chat` | OpenAI Chat Completions, including compatible services | `OPENAI_API_KEY` |

Select a provider with `--policy.api=anthropic --policy.model=...`, for example. For compatible services, also set `--policy.base_url=https://your-provider.example/v1`; `--policy.api_key_env=YOUR_KEY_VARIABLE` selects a different credential variable. Google's custom base URL is the API root, without `/v1beta`.

`--policy.settings` accepts a dictionary of [Pydantic AI model settings](https://ai.pydantic.dev/api/settings/), including native provider settings. OpenAI Responses uses local message history with `store=False`. Transport overrides and server-side conversation IDs are rejected. Provider capabilities still depend on the chosen model; an incompatible model raises an error.

## Control contract

The default config wraps the policy in `StopOnFault | ChunkedSchedule`. A decision sees one frozen observation. It requests a move, the scheduled waypoints play, and the next decision receives measured state again. An accepted target does not prove that the hand arrived: each subsequent observation includes the previous target and remaining translation/gripper error.

Only the task instruction, measured hand pose/gripper, and selected labelled RGB images enter the model prompt. Privileged simulator state and ground-truth success do not. Positions use the measured pose's coordinate frame, in metres. Orientations use roll/pitch/yaw in radians, with `R = Rz(yaw) Ry(pitch) Rx(roll)`. Gripper values run from 0 (open) to 1 (closed).

| Tool | Arguments / effect |
| --- | --- |
| `move_to` | Absolute `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper`, and a short `note` |
| `done` | `reason` and `hindsight`; stop issuing actions for this episode |
| `give_up` | `reason` and `hindsight`; stop issuing actions for this episode |
| `take_pic` | `cameras` and `note`; reveal selected frames in `images=on_demand` mode |

The model must return exactly one tool call. Oversized moves, malformed arguments, unavailable tools, and multiple calls receive explicit correction feedback. Three consecutive invalid replies raise an error. API errors and timeouts surface immediately; SDK retries are disabled. The episode has a budget of 100 calls, including pictures and corrections; exhaustion stops further actions and API calls.

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
| `timeout` | 120 seconds per API call |
| `max_calls` / `max_invalid` | 100 / 3 |
| `camera_keys` | `image.wrist`, `image.exterior` |
| `image_size` | 640 pixels on the longest edge; aspect ratio preserved |
| `image_horizon` | Most recent 2 observations containing images |
| `images` | `always` |

Set, for example, `--policy.motion.max_translation=0.02` or `--policy.images=on_demand`. On-demand pictures reveal frames from the current frozen observation; they do not advance time or move a camera. A camera cannot be revealed twice during one decision. Older images are removed from outgoing history while text, tool results, and native reasoning metadata remain.

Translation is linear and rotation follows the shortest spherical interpolation. A returned move starts from the latest measured pose, with bounds checked again at delivery. A move that no longer fits is discarded with feedback. The driver performs inverse kinematics and enforces its own joint constraints. These bounds are not collision detection or contact-force limits.

## Recordings and cancellation

Each session buffers a compact transcript and exposes a snapshot through `Session.meta`. With `--output_dir`, the episode recorder saves the event list as `inference.policy.transcript` in the episode's `static.json` when the episode finishes. Model configuration, stop reason, and hindsight are stored alongside it in the policy metadata. Without an output directory, the transcript remains in memory.

Events contain the system prompt and tool schemas, measured observations, call numbers, tool replies and text, SDK token usage, rejections, and accepted/discarded decisions. Camera names and observation timestamps refer to the recorded image signals. Images, repeated conversation history, raw HTTP bodies, and provider reasoning signatures are excluded from the transcript. The model's live conversation retains the images and native reasoning metadata needed for subsequent API calls.

The recorded snapshot contains events available when the episode finishes. Later responses and session cleanup do not modify it. Failed or aborted episodes need not retain a transcript.

Only one request can be in flight. Faults and rollout closure invalidate pending decisions; a late response cannot command motion. Closing a rollout cancels the decision before waiting for its current API request, preventing follow-up picture or correction requests during that wait. The runtime waits for the request to finish before the session closes. Cancellation does not promise to stop provider billing for a request already sent.

## Supervised hardware

After validating the task in simulation, use the same policy with `positronic-inference real` and the DROID embodiment. Set the task instruction and output directory explicitly. The keyboard operator controls episode start/stop; the hardware's independent emergency stop remains available. This policy supports the single-arm hand-pose contract, without camera calibration or a workspace/collision model.

## Tests

```bash
uv run --extra llm pytest positronic/vendors/llm/tests
```

Tests mock the actual provider HTTP boundary, check native reasoning/image replay, validate motion and cancellation, and run the full harness/recorder with a deterministic robot. They need no API credentials or hardware.
