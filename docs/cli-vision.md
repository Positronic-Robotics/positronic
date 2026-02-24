# CLI Vision: A Unified `positronic` Command

## 1. Current State

### Entry Points

Positronic ships four separate CLI tools, installed via `pyproject.toml`:

| Tool | Purpose | Config System |
|------|---------|---------------|
| `positronic-data-collection` | Teleop demonstrations in sim or on hardware | configuronic (presets: `sim`, `real`, `so101`, `droid`, `sim_pnp`) |
| `positronic-server` | Web-based dataset browser (Rerun visualization) | configuronic (flat args) |
| `positronic-to-lerobot` | Convert Positronic datasets to LeRobot format | configuronic (subcommands: `convert`, `append`) |
| `positronic-inference` | Run trained policies in sim or on hardware | configuronic (presets: `sim`, `real`, `sim_pnp`) |

Additionally, vendor-specific Docker services provide training and serving:

| Docker Service | Purpose |
|----------------|---------|
| `lerobot-train` | Train ACT models |
| `groot-train` | Train GR00T models |
| `openpi-train` | Train OpenPI (pi0.5) models |
| `openpi-stats` | Compute dataset statistics for OpenPI |
| `lerobot-server` | Serve ACT inference |
| `groot-server` | Serve GR00T inference |
| `openpi-server` | Serve OpenPI inference |

### What Works Well

- **Configuronic** provides composable, type-safe configuration with CLI exposure. Users can override any nested parameter via `--param.nested.key=value`.
- **Preset configs** (`sim`, `real`, `droid`, `so101`) encode complete hardware setups.
- **Codec composition** (`|` operator) decouples data representation from storage.
- **Unified inference protocol** (WebSocket v1) means the same `.remote` policy client works with any vendor server.
- **Docker services** hide vendor-specific training complexity.

### What Doesn't Work Well

- **Fragmented namespace.** Four separate commands with inconsistent naming (`positronic-to-lerobot` is a verb-preposition-noun; the others are `positronic-{noun}`). No discoverability — a new user must read docs to find what commands exist.
- **No unified help.** `positronic --help` doesn't exist. There's no single place to see all capabilities.
- **Training is Docker-only.** The `collect → convert → train → serve → infer` pipeline crosses the boundary between `uv run` and `docker compose run` mid-workflow. This makes scripting and agent automation harder.
- **No structured output.** All commands produce human-readable text. An LLM agent can't reliably parse "what episodes exist" or "what checkpoints are available" from current output.
- **No discovery commands.** Users can't ask "what robots are configured?" or "what models are available?" without reading Python config files.
- **Hardware is hardcoded.** Camera device paths, robot IPs, and servo calibrations live in Python source files (`positronic/cfg/hardware/`). There's no runtime enumeration or user-facing registry.
- **No status/monitoring.** No way to check "what's currently running?" without inspecting processes manually.
- **`cd docker &&` prefix.** Every training/serving command requires changing into the `docker/` directory first, breaking composability and making absolute-path scripts fragile.

---

## 2. Proposed CLI Design

### Design Principles

1. **One command, many subcommands.** `positronic <verb> [noun] [flags]`
2. **Verbs are actions.** `collect`, `train`, `serve`, `run`, `convert`, `view`, `list`, `status`
3. **Nouns are resources.** Robots, models, datasets, servers
4. **Flags before magic.** Explicit is better than inferred, but good defaults mean most flags are optional.
5. **`--json` everywhere.** Every command supports `--json` for structured output.
6. **No `cd` required.** All commands work from any directory.

### Command Tree

```
positronic
├── collect       Collect demonstrations (teleop or scripted)
├── convert       Convert dataset to training format
├── train         Train a policy
├── serve         Start an inference server
├── run           Run a policy on a robot/sim
├── view          Browse and visualize datasets
├── list          Discover available resources
│   ├── robots    Show configured robots/hardware
│   ├── models    Show available model vendors and checkpoints
│   ├── datasets  Show local and remote datasets
│   ├── codecs    Show available codecs
│   └── servers   Show running inference servers
├── status        Show what's running (servers, training jobs, collection sessions)
└── doctor        Check system health (drivers, GPU, Docker, dependencies)
```

### Core Commands

#### `positronic collect` — Collect demonstrations

Replaces `positronic-data-collection`.

```bash
# Minimal: collect in simulation with phone teleop
positronic collect sim --output ~/datasets/stack_cubes

# Explicit robot, explicit teleop device
positronic collect --robot franka --teleop iphone --output ~/datasets/my_task

# With task description (embedded in episode metadata)
positronic collect sim --output ~/data/cubes --task "stack the red cube on the blue cube"

# SO101 bimanual
positronic collect so101 --output ~/datasets/so101_task

# No recording, just test teleop
positronic collect sim --teleop oculus
```

**Key changes from current:**
- Positional preset (`sim`, `real`, etc.) still works for quick starts
- `--robot` flag for explicit robot selection (maps to hardware config)
- `--teleop` instead of `--webxr` (clearer naming, future: keyboard, gamepad)
- `--output` instead of `--output_dir` (shorter, standard)
- All configuronic overrides still work: `--policy.base.checkpoints_dir=...`

#### `positronic convert` — Convert datasets

Replaces `positronic-to-lerobot`.

```bash
# Convert for LeRobot ACT
positronic convert ~/datasets/stack_cubes \
  --model lerobot --codec eepose_absolute \
  --output ~/datasets/lerobot/stack_cubes --fps 30

# Convert for GR00T
positronic convert ~/datasets/stack_cubes \
  --model groot --codec ee_rot6d_joints \
  --output ~/datasets/groot/stack_cubes --fps 15

# Convert for OpenPI
positronic convert ~/datasets/stack_cubes \
  --model openpi --codec eepose \
  --output ~/datasets/openpi/stack_cubes --fps 15

# Append to existing
positronic convert ~/datasets/new_demos \
  --model lerobot --codec eepose_absolute \
  --output ~/datasets/lerobot/stack_cubes --append

# Use public dataset
positronic convert phail:sim_stack_cubes \
  --model lerobot --codec eepose_absolute \
  --output ~/datasets/lerobot/stack_cubes
```

**Key changes:**
- Source dataset is the first positional argument (not `--dataset.dataset.path`)
- `--model` selects vendor (resolves codec module automatically)
- `--codec` is the short name within that vendor (not the full Python path)
- `--append` flag instead of separate `append` subcommand
- Public datasets via `phail:name` shorthand

#### `positronic train` — Train a policy

Wraps Docker training services. This is the most significant new capability — today, training requires manual `docker compose` commands.

```bash
# Train ACT (auto-detects model format from dataset metadata)
positronic train ~/datasets/lerobot/stack_cubes \
  --model lerobot --name baseline_v1 --steps 50000

# Train GR00T
positronic train ~/datasets/groot/stack_cubes \
  --model groot --name groot_v1 --modality ee_rot6d_q

# Train OpenPI (stats computed automatically if missing)
positronic train ~/datasets/openpi/stack_cubes \
  --model openpi --name openpi_v1

# Resume training
positronic train ~/datasets/lerobot/stack_cubes \
  --model lerobot --name baseline_v1 --resume

# Train on remote machine
positronic train ~/datasets/lerobot/stack_cubes \
  --model lerobot --name baseline_v1 --machine train
```

**Key changes:**
- Hides Docker complexity behind a single command
- `--machine` flag to dispatch to remote training infrastructure (from `/remote-training` skill)
- Auto-generates OpenPI stats if not present
- Dataset path is positional

#### `positronic serve` — Start an inference server

Wraps Docker inference servers.

```bash
# Serve latest checkpoint
positronic serve ~/checkpoints/lerobot/baseline_v1 \
  --model lerobot --codec eepose_absolute

# Serve specific checkpoint on specific port
positronic serve ~/checkpoints/groot/groot_v1 \
  --model groot --codec ee_rot6d_joints \
  --checkpoint 20000 --port 8001

# Serve OpenPI
positronic serve ~/checkpoints/openpi/openpi_v1 \
  --model openpi --codec eepose

# JSON output for agents
positronic serve ~/checkpoints/lerobot/baseline_v1 \
  --model lerobot --codec eepose_absolute --json
# → {"status": "running", "port": 8000, "model_id": "50000", "url": "ws://localhost:8000/api/v1/session"}
```

#### `positronic run` — Run inference

Replaces `positronic-inference`.

```bash
# Run in sim with remote server (most common)
positronic run sim --server localhost:8000

# Run in sim with local ACT policy
positronic run sim --policy act --checkpoint ~/checkpoints/lerobot/baseline_v1

# Run on real Franka
positronic run real --server gpu-box:8000

# Batch evaluation: 10 episodes, 30s each, record results
positronic run sim --server localhost:8000 \
  --episodes 10 --duration 30 \
  --output ~/datasets/eval/baseline_v1

# Keyboard control for debugging
positronic run sim --server localhost:8000 --driver keyboard

# JSON output for agent parsing
positronic run sim --server localhost:8000 --episodes 5 --json
# → {"episodes": [{"id": 0, "duration": 15.2, "status": "completed"}, ...]}
```

**Key changes:**
- `--server host:port` replaces `--policy=.remote --policy.host=... --policy.port=...`
- `--episodes` and `--duration` replace `--driver.num_iterations` and `--driver.simulation_time`
- `--driver keyboard` replaces `--driver=.keyboard`

#### `positronic view` — Browse datasets

Replaces `positronic-server`.

```bash
# View a dataset
positronic view ~/datasets/stack_cubes

# View on custom port
positronic view ~/datasets/stack_cubes --port 5001

# View inference logs
positronic view ~/datasets/eval/baseline_v1

# List episodes (for agents)
positronic view ~/datasets/stack_cubes --json
# → {"episodes": 42, "total_duration": "12m30s", "signals": ["robot_state", "left_cam", ...]}
```

#### `positronic list` — Discover resources

New command family for discoverability. This is critical for agent-friendliness.

```bash
# List configured robots
positronic list robots
# → franka    Franka Panda (172.168.0.2)
# → kinova    Kinova Gen3 (192.168.1.10)
# → so101     SO101 Bimanual (Feetech /dev/ttyACM0)
# → sim       MuJoCo Franka (built-in)

positronic list robots --json
# → [{"name": "franka", "type": "FrankaPanda", "ip": "172.168.0.2", "status": "unknown"}, ...]

# List available models/vendors
positronic list models
# → lerobot   LeRobot ACT          Local GPU (4GB+)     Single-task, fast iteration
# → groot     NVIDIA GR00T         GPU (7.5GB inference) Generalist
# → openpi    OpenPI (pi0.5)       GPU (62GB inference)  Most capable

# List checkpoints for a model
positronic list models lerobot --checkpoints-dir ~/checkpoints/lerobot/
# → baseline_v1/  checkpoints: [10000, 20000, 30000, 40000, 50000]
# → finetune_v2/  checkpoints: [5000, 10000]

# List local datasets
positronic list datasets ~/datasets/
# → stack_cubes/          42 episodes  12m30s  raw
# → lerobot/stack_cubes/  42 episodes  12m30s  lerobot (eepose_absolute)

# List available codecs for a model
positronic list codecs lerobot
# → eepose_absolute        EE pose (7D) + grip → absolute position
# → joints_absolute        Joint positions (7D) + grip → absolute position
# → eepose_absolute_traj   EE pose → trajectory (actual robot path)
# → joints_absolute_traj   Joint positions → trajectory

positronic list codecs groot
# → ee_absolute            EE pose (quat) → absolute (modality: ee)
# → ee_rot6d               EE pose (rot6d) → absolute (modality: ee_rot6d)
# → ee_joints              EE pose + joints → absolute (modality: ee_q)
# → ee_rot6d_joints        EE pose (rot6d) + joints → absolute (modality: ee_rot6d_q)

# List running servers
positronic list servers
# → localhost:8000  lerobot  baseline_v1/50000  ready
# → gpu-box:8000   openpi   openpi_v1/latest   loading (45%)
```

#### `positronic status` — Show what's running

```bash
positronic status
# Servers:
#   localhost:8000  lerobot baseline_v1  checkpoint 50000  ready
#
# Training:
#   groot_v1  groot  step 12000/50000 (24%)  loss: 0.023  ETA: 3h
#
# Collection:
#   (none active)

positronic status --json
# → {"servers": [...], "training": [...], "collection": [...]}
```

#### `positronic doctor` — System health check

```bash
positronic doctor
# ✓ Python 3.11.8
# ✓ uv 0.5.1
# ✓ Docker 27.0.3 (daemon running)
# ✓ GPU: NVIDIA RTX 4090 (CUDA 12.4)
# ✓ MuJoCo 3.2.0
# ✗ Franka: 172.168.0.2 unreachable
# ✓ ZED SDK 4.0
# ✓ Rerun 0.22.1
```

### Pipeline Examples

#### "Pick a model, pick a robot, run"

The core Positronic promise in three commands:

```bash
# 1. Collect 50 demos
positronic collect sim --output ~/data/cubes --task "stack cubes"

# 2. Train (picks codec automatically based on model defaults)
positronic convert ~/data/cubes --model lerobot --output ~/data/lerobot/cubes
positronic train ~/data/lerobot/cubes --model lerobot --name cubes_v1

# 3. Run
positronic serve ~/checkpoints/lerobot/cubes_v1 --model lerobot --codec eepose_absolute &
positronic run sim --server localhost:8000 --episodes 10
```

#### Switching models

```bash
# Same raw data, different model
positronic convert ~/data/cubes --model groot --codec ee_rot6d_joints --output ~/data/groot/cubes
positronic train ~/data/groot/cubes --model groot --name cubes_groot_v1
positronic serve ~/checkpoints/groot/cubes_groot_v1 --model groot --codec ee_rot6d_joints --port 8001 &
positronic run sim --server localhost:8001 --episodes 10
```

#### Switching robots

```bash
# Same policy, different robot
positronic run real --server gpu-box:8000         # Franka
positronic run so101 --server gpu-box:8000        # SO101
```

---

## 3. Agent Interaction Examples

These show how Claude Code (or any LLM agent) would use the CLI to accomplish real robotics tasks.

### "Run ACT on the Franka sim"

```
User: Run ACT on the Franka sim

Agent thinks: Need to find an ACT checkpoint and start inference in sim.

$ positronic list models lerobot --checkpoints-dir ~/checkpoints/lerobot/ --json
→ [{"experiment": "baseline_v1", "checkpoints": [10000, 20000, 30000]}]

$ positronic serve ~/checkpoints/lerobot/baseline_v1 \
    --model lerobot --codec eepose_absolute --json
→ {"status": "running", "port": 8000, "model_id": "30000"}

$ positronic run sim --server localhost:8000 --episodes 3 --json
→ {"episodes": [
    {"id": 0, "duration": 14.8, "status": "completed"},
    {"id": 1, "duration": 15.0, "status": "completed"},
    {"id": 2, "duration": 14.6, "status": "completed"}
  ]}

Agent: Ran 3 episodes of ACT inference in MuJoCo sim. All completed
successfully with ~15s duration each. The server is running on
localhost:8000 with checkpoint 30000.
```

### "Switch to SO101 and re-run"

```
User: Switch to SO101 and re-run

Agent thinks: Server is already running from previous task. Just need
to switch the robot target.

$ positronic run so101 --server localhost:8000 --episodes 3 --json
→ {"episodes": [
    {"id": 0, "duration": 12.1, "status": "completed"},
    {"id": 1, "duration": 11.8, "status": "completed"},
    {"id": 2, "duration": 12.3, "status": "completed"}
  ]}

Agent: Ran 3 episodes on the SO101 robot using the same ACT policy.
Episodes completed with ~12s duration.
```

### "Collect 10 demos and train"

```
User: Collect 10 demos of stacking cubes in sim, then train ACT

Agent thinks: Multi-step pipeline — collect, convert, train.

$ positronic collect sim \
    --output ~/datasets/cubes_$(date +%Y%m%d) \
    --task "stack the red cube on the blue cube" \
    --teleop iphone
→ (interactive — user collects demos via phone)
→ Recorded 10 episodes to ~/datasets/cubes_20260222

$ positronic view ~/datasets/cubes_20260222 --json
→ {"path": "~/datasets/cubes_20260222", "episodes": 10,
   "total_duration": "3m42s", "signals": ["robot_state", "left_cam", "right_cam"]}

$ positronic convert ~/datasets/cubes_20260222 \
    --model lerobot --codec eepose_absolute \
    --output ~/datasets/lerobot/cubes_20260222 --fps 30 --json
→ {"status": "completed", "episodes_converted": 10,
   "output": "~/datasets/lerobot/cubes_20260222"}

$ positronic train ~/datasets/lerobot/cubes_20260222 \
    --model lerobot --name cubes_baseline --steps 50000 --json
→ {"status": "completed", "experiment": "cubes_baseline",
   "checkpoints": [10000, 20000, 30000, 40000, 50000],
   "final_loss": 0.018, "checkpoints_dir": "~/checkpoints/lerobot/cubes_baseline"}

Agent: Collected 10 demos (3m42s total), converted to LeRobot format
at 30fps, and trained ACT for 50k steps. Final loss: 0.018.
Checkpoints saved to ~/checkpoints/lerobot/cubes_baseline.
Ready to serve with:
  positronic serve ~/checkpoints/lerobot/cubes_baseline --model lerobot --codec eepose_absolute
```

### "What models are available for this robot?"

```
User: What models can I run on the SO101?

$ positronic list models --json
→ [
    {"name": "lerobot", "display": "LeRobot ACT",
     "inference_gpu": "4GB+", "training_gpu": "16GB+",
     "description": "Single-task, fast iteration"},
    {"name": "groot", "display": "NVIDIA GR00T",
     "inference_gpu": "7.5GB", "training_gpu": "50GB",
     "description": "Generalist foundation model"},
    {"name": "openpi", "display": "OpenPI (pi0.5)",
     "inference_gpu": "62GB", "training_gpu": "78GB",
     "description": "Most capable foundation model"}
  ]

$ positronic list codecs lerobot --json
→ [{"name": "eepose_absolute", "obs": "EE pose + grip", "action": "absolute position"}, ...]

$ positronic list codecs groot --json
→ [{"name": "ee_rot6d_joints", "obs": "EE pose (rot6d) + joints + grip", ...}, ...]

Agent: All three models work with the SO101:

| Model | Inference GPU | Best For |
|-------|--------------|----------|
| LeRobot ACT | 4GB+ | Fast iteration, single task |
| GR00T | 7.5GB | Generalist, can run on robot GPU |
| OpenPI | 62GB | Most capable, needs cloud GPU |

Recommendation: Start with LeRobot ACT for fast iteration,
then try GR00T if you need multi-task generalization.
```

### "Check on my training job"

```
User: How's training going?

$ positronic status --json
→ {
    "servers": [],
    "training": [{
      "name": "cubes_baseline",
      "model": "lerobot",
      "step": 35000,
      "total_steps": 50000,
      "loss": 0.021,
      "dataset": "~/datasets/lerobot/cubes_20260222"
    }],
    "collection": []
  }

Agent: Training is 70% complete (step 35000/50000). Current loss: 0.021.
```

---

## 4. Gap Analysis

### Command-by-command assessment

#### `positronic collect` (effort: Low)

| Aspect | Status | Notes |
|--------|--------|-------|
| Core functionality | **Exists** | `positronic-data-collection` does this today |
| Configuronic CLI | **Exists** | Presets and overrides work |
| Hardware presets | **Exists** | `sim`, `real`, `so101`, `droid`, `sim_pnp` |
| Teleop (WebXR) | **Exists** | iPhone, Android, Oculus all work |
| Recording to dataset | **Exists** | `LocalDatasetWriter` is production-quality |
| `--json` output | **Missing** | Needs structured output for episode results |
| Cleaner flag names | **Missing** | `--teleop` vs `--webxr`, `--output` vs `--output_dir` |

**Gap:** Mostly a renaming/aliasing exercise. The hard work (pimm orchestration, WebXR, recording) is done.

#### `positronic convert` (effort: Low)

| Aspect | Status | Notes |
|--------|--------|-------|
| Core conversion | **Exists** | `positronic-to-lerobot` works for all three vendors |
| Codec system | **Exists** | Composable, lazy, production-quality |
| Parallel processing | **Exists** | DataLoader with configurable workers |
| `--model` shorthand | **Missing** | Currently requires full Python path for codec |
| `--append` flag | **Partially exists** | Separate `append` subcommand exists |
| `--json` output | **Missing** | Needs structured output |
| Public dataset shorthand | **Missing** | `phail:sim_stack_cubes` syntax |

**Gap:** Mostly UX polish. A codec registry that maps `(vendor, short_name)` → full codec path would make this much cleaner.

#### `positronic train` (effort: Medium)

| Aspect | Status | Notes |
|--------|--------|-------|
| LeRobot training | **Exists** | Docker service `lerobot-train` |
| GR00T training | **Exists** | Docker service `groot-train` |
| OpenPI training | **Exists** | Docker service `openpi-train` + `openpi-stats` |
| Auto stats for OpenPI | **Missing** | Currently separate manual step |
| Unified CLI wrapper | **Missing** | Today requires `cd docker && docker compose run` |
| `--machine` dispatch | **Partially exists** | `/remote-training` skill handles VMs |
| Progress monitoring | **Missing** | No structured progress output from Docker |
| `--json` output | **Missing** | Training output is unstructured logs |

**Gap:** Medium effort. Need a Python wrapper that:
1. Resolves vendor from dataset or `--model` flag
2. Invokes the correct Docker Compose service
3. Pipes stdout for progress
4. Auto-runs `openpi-stats` when needed

The hard part is reliable progress parsing from Docker output streams, and clean handling of Docker Compose from any working directory.

#### `positronic serve` (effort: Medium)

| Aspect | Status | Notes |
|--------|--------|-------|
| Vendor servers | **Exist** | All three vendor servers work via Docker |
| WebSocket protocol | **Exists** | Protocol v1 is unified |
| `/api/v1/models` endpoint | **Exists** | Lists available checkpoints |
| Unified CLI wrapper | **Missing** | Today requires `cd docker && docker compose run` |
| Server registration/tracking | **Missing** | No central registry of running servers |
| `--json` output | **Missing** | Startup output is unstructured |
| Health check endpoint | **Partially exists** | Servers have `/api/v1/models` but no `/health` |

**Gap:** Similar to `train` — need a wrapper around Docker Compose. Server registration (for `positronic list servers` and `positronic status`) needs a lightweight daemon or PID file mechanism.

#### `positronic run` (effort: Low)

| Aspect | Status | Notes |
|--------|--------|-------|
| Core inference loop | **Exists** | `positronic-inference` does this today |
| Remote policy | **Exists** | `RemotePolicy` connects to any vendor server |
| Local policy (ACT) | **Exists** | `LerobotPolicy` loads from checkpoint |
| Sim environments | **Exists** | MuJoCo with multiple scenes |
| Hardware environments | **Exists** | Franka, Kinova, SO101 |
| Recording | **Exists** | `LocalDatasetWriter` during inference |
| Batch evaluation | **Exists** | Timed driver with iteration count |
| `--server host:port` | **Missing** | Currently `--policy=.remote --policy.host=... --policy.port=...` |
| `--json` output | **Missing** | Episode results not structured |
| `--episodes` / `--duration` | **Missing** | Currently `--driver.num_iterations` / `--driver.simulation_time` |

**Gap:** Mostly flag aliasing and structured output. The inference machinery is complete.

#### `positronic view` (effort: Low)

| Aspect | Status | Notes |
|--------|--------|-------|
| Web server | **Exists** | `positronic-server` with Rerun visualization |
| Episode browsing | **Exists** | Sortable, filterable table |
| RRD generation | **Exists** | On-demand with caching |
| `--json` output | **Missing** | Dataset summary not available as JSON |
| Rename to `view` | **Trivial** | Just an entry point rename |

**Gap:** Minimal. Rename + add `--json` for dataset summary.

#### `positronic list` (effort: Medium-High)

| Aspect | Status | Notes |
|--------|--------|-------|
| Robot configs exist | **Yes** | In `positronic/cfg/hardware/` as Python objects |
| Robot registry/enumeration | **Missing** | No API to list "what robots are configured" |
| Model vendor list | **Hardcoded** | Three vendors, could be a simple list |
| Checkpoint enumeration | **Partially exists** | Servers can list checkpoints; no standalone tool |
| Dataset enumeration | **Missing** | No tool to scan a directory for datasets |
| Codec enumeration | **Missing** | Codecs are Python objects, not a registry |
| Running server discovery | **Missing** | No server registration mechanism |

**Gap:** This is the biggest new feature. Requires:
1. A **robot registry** — either scan `positronic/cfg/hardware/` via introspection or maintain a config file.
2. A **codec registry** — map `(vendor, short_name)` to codec objects. Could be derived from module introspection.
3. A **dataset scanner** — walk directories looking for `meta.json` / `static.json` markers.
4. A **server tracker** — PID files, a lightweight daemon, or just probing known ports.

#### `positronic status` (effort: Medium)

| Aspect | Status | Notes |
|--------|--------|-------|
| Server status | **Missing** | No central tracking of running servers |
| Training job status | **Missing** | No tracking of Docker training containers |
| Collection session status | **Missing** | No tracking of active data collection |

**Gap:** Requires process/container tracking. Could be implemented via:
- Docker API for training containers
- PID files + port probing for servers
- PID files for collection sessions

#### `positronic doctor` (effort: Low-Medium)

| Aspect | Status | Notes |
|--------|--------|-------|
| Python version check | **Trivial** | `sys.version` |
| GPU detection | **Partially exists** | `torch.cuda.is_available()` used internally |
| Docker check | **Trivial** | `docker info` |
| Hardware connectivity | **Missing** | No ping/probe for configured robots |
| Dependency check | **Exists** | `uv` handles this, but could be surfaced |

**Gap:** Straightforward to implement. Most checks are one-liners.

### Summary Scorecard

| Command | Functionality Exists | CLI Wrapper Needed | Structured Output | Effort |
|---------|---------------------|-------------------|-------------------|--------|
| `collect` | Yes | Rename/alias | Add `--json` | **Low** |
| `convert` | Yes | Codec registry | Add `--json` | **Low** |
| `train` | Yes (Docker) | Docker wrapper | Add `--json` | **Medium** |
| `serve` | Yes (Docker) | Docker wrapper | Add `--json` | **Medium** |
| `run` | Yes | Flag aliasing | Add `--json` | **Low** |
| `view` | Yes | Rename | Add `--json` | **Low** |
| `list` | Partial | New registries | Native JSON | **Medium-High** |
| `status` | No | New feature | Native JSON | **Medium** |
| `doctor` | No | New feature | Native JSON | **Low-Medium** |

**Overall distance:** The core robotics functionality (collect, convert, train, serve, run) is all built and working. What's missing is:
1. A unified entry point with subcommand dispatch
2. Structured JSON output on every command
3. Resource registries for discovery (`list`)
4. Process tracking for monitoring (`status`)

The codebase is **close** — maybe 70% of the way there for the core commands, 30% for the discovery/monitoring layer.

---

## 5. Suggested Implementation Order

### Phase 1: Unified entry point and low-effort commands

**Build the `positronic` command with subcommand dispatch.** Wire up the easy commands first to establish the pattern.

1. **Create `positronic` entry point** with click/typer subcommand dispatch. Each subcommand delegates to existing configuronic CLI machinery where possible.
2. **`positronic view`** — Rename of `positronic-server`. Trivial.
3. **`positronic collect`** — Thin wrapper around `positronic-data-collection` with cleaner flag names.
4. **`positronic run`** — Thin wrapper around `positronic-inference` with `--server host:port` convenience flag.
5. **`positronic convert`** — Wrapper with codec registry for `--model vendor --codec short_name` syntax.

**Why first:** Gets the `positronic` command into users' hands immediately. All underlying functionality already works. Teaches the team the pattern for adding new subcommands.

### Phase 2: Docker wrappers

6. **`positronic train`** — Python wrapper that invokes `docker compose run` for the correct vendor service. Handles `cd docker` internally.
7. **`positronic serve`** — Same pattern. Write server PID/port to a tracking file on startup.
8. **`positronic doctor`** — System health checks.

**Why second:** These remove the biggest UX friction (the `cd docker && docker compose run` pattern) and make the full pipeline scriptable.

### Phase 3: Discovery and monitoring

9. **Codec registry** — Introspect vendor modules to build `(vendor, name) → codec` mapping. Enables `positronic list codecs`.
10. **Dataset scanner** — Walk directories for dataset markers. Enables `positronic list datasets`.
11. **`positronic list`** subcommands — robots, models, codecs, datasets, servers.
12. **`positronic status`** — Read tracking files, probe Docker, report state.

**Why third:** These are the agent-friendliness features. They require new infrastructure (registries, scanners, trackers) but don't block human workflows.

### Phase 4: JSON output everywhere

13. **Add `--json` flag** to every command. Each command's output should have a well-defined JSON schema.

**Why it can wait:** JSON output is important for agents but can be added incrementally to each command. The commands themselves are useful without it.

---

## 6. Open Questions

### CLI framework choice

Configuronic currently handles CLI parsing for all entry points. A unified `positronic` command needs subcommand dispatch, which configuronic doesn't provide natively. Options:
- **click/typer for top-level dispatch**, delegate to configuronic for each subcommand's flags
- **Extend configuronic** to support subcommand trees
- **Replace configuronic** with click/typer entirely (high effort, breaks existing workflow)

Recommendation: click or typer for the top-level `positronic` command, with each subcommand calling into configuronic's `cfn.cli()` for the complex configuration. This preserves all existing deep-override capability while adding discoverability at the top level.

### How much Docker to hide?

`positronic train` and `positronic serve` would wrap Docker Compose. Questions:
- Should the CLI also handle Docker image building/pulling?
- What happens when Docker isn't installed? Fall back to local execution?
- How to pass through GPU allocation, environment variables, volume mounts?
- Should we support non-Docker training (local `uv run` for LeRobot ACT which doesn't need heavy dependencies)?

### Robot configuration: code vs config file?

Currently, robots are Python objects in `positronic/cfg/hardware/`. For `positronic list robots` to work, we need either:
- **Module introspection** — scan the hardware config module for all `@cfn.config()` decorated functions
- **A config file** — `~/.positronic/robots.yaml` or similar, listing configured robots
- **Convention** — each hardware config module exports a `REGISTRY` dict

A config file is more agent-friendly (can be read/written without Python import) but adds a new source of truth that could drift from the code.

### Codec short names

The `--codec eepose_absolute` shorthand requires a mapping from `(vendor, short_name)` to the full Python codec path. Where should this live?
- In each vendor module (e.g., `positronic/vendors/lerobot/codecs.py` exports `REGISTRY`)
- In a central registry file
- Derived by introspection at runtime

### Server process management

`positronic serve` starts a Docker container. How should the CLI track it?
- **PID files** in `~/.positronic/servers/`
- **Docker labels** — tag containers with metadata, query via Docker API
- **Port scanning** — just probe known ports
- **Lightweight daemon** — overkill for now

Docker labels seem most natural since the servers already run in Docker.

### Backward compatibility

Should the old commands (`positronic-data-collection`, `positronic-server`, etc.) continue to work?
- **Yes (recommended initially)** — keep old entry points as aliases, add deprecation warnings
- **No** — clean break, update all docs at once

### Training on remote machines

The `/remote-training` skill already manages Nebius VMs. Should `positronic train --machine train` integrate with this, or should remote training remain a separate workflow?

### Default codec per vendor

When a user runs `positronic convert --model lerobot`, should there be a default codec so `--codec` can be omitted? If so, what are the defaults?
- LeRobot: `eepose_absolute` (most common in examples)
- GR00T: `ee_rot6d_joints` (recommended in docs)
- OpenPI: `eepose` (most common)

### Compositional pipelines

Should the CLI support piping? Example:
```bash
positronic collect sim --output - | positronic convert --model lerobot --output ~/data/lr
```

This would require streaming dataset output, which is a significant architectural change. Probably not worth it — the commands are naturally sequential with filesystem as the interface.

---

## See Also

- [Model Selection](model-selection.md) — Choosing models
- [Training Workflow](training-workflow.md) — Current end-to-end workflow
- [Codecs Guide](codecs.md) — Understanding codecs
- [Inference Guide](inference.md) — Deployment patterns
- [Data Collection Guide](data-collection.md) — Teleoperation workflow
