# Positronic — Python-native stack for real-life ML robotics

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://github.com/Positronic-Robotics/positronic/actions/workflows/unit-test.yaml/badge.svg)](https://github.com/Positronic-Robotics/positronic/actions/workflows/unit-test.yaml)
[![PyPI](https://img.shields.io/pypi/v/positronic)](https://pypi.org/project/positronic/)
[![Discord](https://img.shields.io/badge/Discord-Positronic%20Robotics-5865F2?logo=discord&logoColor=white)](https://discord.gg/PXvBy4NBgv)

## The Problem

AI promises to transform robotics: teach robots through demonstrations instead of code. ML-driven approaches can unlock capabilities traditional analytical control can't reach.

The field is early. The ecosystem lacks dedicated instruments to make development simple, repeatable, and accessible:

1. **Data collection is expensive**: Hardware integration, teleoperation setup, dataset curation all require specialized expertise
2. **Data is messy**: Multi-rate sensors, format fragmentation, re-recording for each framework, datasets thrown away when you try different state/action representations
3. **Deployment is complex**: Vendor-specific APIs, hardware compatibility issues, monitoring infrastructure from scratch

Positronic solves these operational challenges so teams building manipulation systems can focus on what their robots should do, not how to make the infrastructure work.

## What is Positronic

Positronic is an end-to-end toolkit for building ML-driven robotics systems.

It covers the full lifecycle: bring new hardware online, capture and curate datasets, train and evaluate policies, deploy inference, monitor performance, and iterate when behaviour drifts.

Every subsystem is implemented in plain Python. No ROS required. Compatible with LeRobot training and foundation models like OpenPI and GR00T.

**Our goal is to make professional-grade ML robotics approachable. Join the conversation on the [Positronic Discord](https://discord.gg/PXvBy4NBgv) to share feedback, showcase projects, and get help from the community.**

> **Positronic is under heavy development and in alpha stage. APIs, interfaces, and workflows may change significantly between releases.**

## Why Positronic

### Standing on Giants' Shoulders

Positronic builds on the robotics ML ecosystem:
- [LeRobot/HuggingFace](https://github.com/huggingface/lerobot) for training scripts and workflows
- [MuJoCo](https://mujoco.org) for physics simulation
- [Rerun.io](https://rerun.io) for visualization
- Foundation model builders: [Physical Intelligence](https://www.physicalintelligence.company/) and [NVIDIA](https://developer.nvidia.com/isaac/groot)

We focus on what's missing: the plumbing, hardware integration, and operational lifecycle that production systems need.

**The ecosystem provides**: Training frameworks, foundation models, simulation engines, model research
**Positronic adds**: Data ops, hardware drivers, unified inference API, iteration workflows, deployment infrastructure

### Store Once, Use Everywhere: Dataset Library

**Problem solved**: Stop re-recording datasets for each framework AND stop throwing away datasets when you want different state/action formats.

The [Positronic dataset library](positronic/dataset/README.md) provides raw data storage and a unified API for plumbing, preprocessing, and backward compatibility. Codecs apply lazy transforms to convert one dataset into LeRobot, GR00T, or OpenPI format without re-recording.

Try different state representations (joint space vs end-effector space), action formats (absolute vs delta), observation encodings, all from the same raw data. Immutable storage, composable transforms, infinite uses.

### Connect ANY Hardware to ANY Model: Unified Inference API

**Problem solved**: Vendor lock-in and API fragmentation.

The [offboard inference system](positronic/offboard/README.md) provides a single WebSocket protocol (v1) across all vendors. The `RemotePolicy` client works interchangeably with LeRobot, GR00T, and OpenPI servers.

Built-in status streaming handles long model loads (120-300s) gracefully. Swap models without changing hardware code.

### Immediate-Mode Runtime (Pimm)

[`pimm`](pimm/README.md) wires sensors, controllers, inference, and GUIs without ROS launch files or bespoke DSLs. Control loops stay testable and readable. See the [Pimm README](pimm/README.md) for details.

## Foundation Models — Choose by Capability

Positronic supports state-of-the-art foundation models with first-class workflows:

| Model | Capability | Training | Inference | Best For |
|-------|-----------|----------|-----------|----------|
| **[OpenPI (π₀.₅)](positronic/vendors/openpi/README.md)** | Most capable, generalist | Capable GPU (~78GB, LoRA) | Capable GPU (~62GB) | Complex multi-task manipulation |
| **[GR00T](positronic/vendors/gr00t/README.md)** | Generalist robot policy | Capable GPU (~50GB) | Smaller GPU (~7.5GB) | Logistics and industry applications |
| **[SmolVLA](positronic/vendors/lerobot/README.md)** | Vision-language-action | Consumer GPU | Consumer GPU | Language-conditioned manipulation |
| **[LeRobot ACT](positronic/vendors/lerobot_0_3_3/README.md)** | Single-task, efficient | Consumer GPU | Consumer GPU | Specific manipulation tasks |

**Recommendation**: Start with SmolVLA or ACT if you want something quick and low-cost. Progress to GR00T or OpenPI if you need more capable models. Positronic makes switching easy.

### Why Multiple Vendors?

Our goal is to **democratize ML/AI in robotics**. You shouldn't be locked to a single vendor or architecture.

Positronic's plug-and-play structure means:
- **Same dataset format** — Record once, train on any model
- **Same inference API** — Swap models without changing hardware code
- **Easy experimentation** — Try all models with your data, pick what works best
- **Future-proof** — We'll keep adding foundation models as they emerge

See [Model Selection Guide](docs/model-selection.md) for detailed comparison and decision criteria.

## Installation

Clone the repository and set up a local `uv` environment.

### Local Installation via uv

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), `libturbojpeg`, and FFmpeg
```
sudo apt install libturbojpeg ffmpeg portaudio19-dev  # Linux
brew install jpeg-turbo ffmpeg                        # macOS
```

```bash
git clone git@github.com:Positronic-Robotics/positronic.git
cd positronic

uv venv -p 3.11               # optional but keeps the interpreter isolated
source .venv/bin/activate     # activate the venv if you created one
uv sync --frozen --extra dev  # install core + dev tooling
```

Install hardware extras only when you need physical robot drivers (Linux only):

```bash
uv sync --frozen --extra hardware
```

After installation, the following command-line scripts will be available:
- `positronic-data-collection`: Collect demonstrations in simulation or on hardware
- `positronic-server`: Browse and inspect datasets
- `lerobot-convert` (Docker): Convert datasets to model format
- `positronic-inference`: Run trained policies in simulation or on hardware

All commands work both inside an activated virtual environment and with `uv run` prefix (e.g., `uv run positronic-server`).

For training and inference servers, use vendor-specific Docker services (see [Training Workflow](docs/training-workflow.md)).

## Quick Start — 30 Seconds to Data Collection

```bash
uv run positronic-data-collection sim \
    --output_dir=~/datasets/stack_cubes_raw \
    --sound=None --webxr=.iphone
```

Opens MuJoCo simulation with phone-based teleoperation. Record demonstrations by moving your phone to control the robot arm and using the on-screen controls to open/close the gripper and start/stop recording.

Then browse your episodes:

```bash
uv run positronic-server --dataset.path=~/datasets/stack_cubes_raw --port=5001
```

Visit `http://localhost:5001` to view episodes. Continue to full workflow below.

## End-to-End Workflow

The usual loop is: collect demonstrations → review and curate → train → validate and iterate.

### 1. Collect Demonstrations

Use the [data collection script](positronic/data_collection.py) for both simulation and hardware captures.

**Quick start in simulation:**

```bash
uv run positronic-data-collection sim \
    --output_dir=~/datasets/stack_cubes_raw \
    --sound=None --webxr=.iphone --operator_position=.BACK
```

Loads the MuJoCo scene, starts the DearPyGui UI, and records episodes into the local dataset.

**Teleoperation:**
- Phone (iPhone/Android) or VR headset (Oculus) control the robot in 6-DOF
- Browser shows AR interface with Track, Record, Reset buttons
- See [Data Collection Guide](docs/data-collection.md) for complete setup

**Physical robots:**

```bash
uv run positronic-data-collection real  --output_dir=~/datasets/franka_kitchen
uv run positronic-data-collection so101 --output_dir=~/datasets/so101_runs
uv run positronic-data-collection droid --output_dir=~/datasets/droid_runs
```

### 2. Review and Curate

Browse datasets with the [positronic-server](positronic/server/positronic_server.py):

```bash
uv run positronic-server \
    --dataset.path=~/datasets/stack_cubes_raw \
    --port=5001
```

Visit `http://localhost:5001` to view episodes. The viewer is read-only for now: mark low-quality runs while watching, then rename or remove the corresponding episode directories manually.

To preview exactly what the training will see, pass the same codec configuration you'll use for conversion:

```bash
uv run positronic-server \
    --dataset=@positronic.cfg.ds.local_all \
    --dataset.path=~/datasets/stack_cubes_raw \
    --dataset.codec=@positronic.vendors.openpi.codecs.ee \
    --port=5001
```

### 3. Prepare Data for Training

Convert curated runs using a codec:

```bash
cd docker && docker compose run --rm lerobot-convert convert \
    --dataset.dataset=.local \
    --dataset.dataset.path=~/datasets/stack_cubes_raw \
    --dataset.codec=@positronic.vendors.lerobot.codecs.ee \
    --output_dir=~/datasets/lerobot/stack_cubes \
    --task="pick up the green cube and place it on the red cube"
```

**Train using vendor-specific workflows:**

Training is handled through Docker services. Example with ACT (fastest baseline):

```bash
cd docker && docker compose run --rm lerobot-train \
    --input_path=~/datasets/lerobot/stack_cubes \
    --exp_name=stack_cubes_act \
    --output_dir=~/checkpoints/lerobot/
```

Progress to OpenPI or GR00T when you need more capable models. See:
- [Training Workflow Guide](docs/training-workflow.md)
- [Model Selection Guide](docs/model-selection.md)
- [Codec Selection Guide](docs/codecs.md)

### 4. Run Inference and Iterate

Run trained policies through the [inference script](positronic/inference.py):

```bash
uv run positronic-inference sim \
    --policy=@positronic.cfg.policy.openpi_absolute \
    --policy.base.checkpoints_dir=~/checkpoints/openpi/<run_id> \
    --driver.simulation_time=60 \
    --driver.show_gui=True \
    --output_dir=~/datasets/inference_logs/stack_cubes_pi0
```

**Remote inference** (run policy on a different machine):

```bash
# On inference server:
cd docker && docker compose run --rm --service-ports lerobot-server \
    --checkpoints_dir=~/checkpoints/lerobot/<run_id> \
    --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee

# On robot:
uv run positronic-inference sim \
    --policy=.remote \
    --policy.host=<server-ip>
```

Monitor performance, collect edge cases, and iterate. See [Inference Guide](docs/inference.md) for details.

## Documentation

**Core Concepts:**
- [Dataset Library](positronic/dataset/README.md) — Storage, codecs, transforms
- [Pimm Runtime](pimm/README.md) — Immediate-mode control systems
- [Offboard Inference](positronic/offboard/README.md) — Unified protocol

**Model Workflows:**
- [OpenPI (π₀.₅)](positronic/vendors/openpi/README.md) — Recommended for most tasks
- [GR00T](positronic/vendors/gr00t/README.md) — NVIDIA's generalist policy
- [SmolVLA / LeRobot 0.4.x](positronic/vendors/lerobot/README.md) — Vision-language-action
- [LeRobot ACT](positronic/vendors/lerobot_0_3_3/README.md) — Single-task transformer

**Guides:**
- [Model Selection](docs/model-selection.md) | [Codecs](docs/codecs.md) | [Training](docs/training-workflow.md)
- [Data Collection](docs/data-collection.md) | [Inference](docs/inference.md)

**Hardware:**
- [Drivers](positronic/drivers/) — Robot arms, cameras, grippers
- [Hardware Configs](positronic/cfg/hardware/) — Franka, Kinova, SO101, DROID

## Development workflow

Install development dependencies first:
```bash
uv sync --frozen --extra dev  # install core + dev tooling
```

### Initial Setup

Install pre-commit hooks (one-time setup):

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type post-commit
```

### Daily Development

Run tests and linters from the root directory:

```bash
uv run pytest --no-cov
uv run ruff check .
uv run ruff format .
```

Use `uv add` / `uv remove` to modify dependencies and `uv lock` to refresh the lockfile.

### Contributing

We welcome contributions from the community! Before submitting a pull request, please:

1. Read our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines
2. Sign your commits cryptographically (SSH or GPG signing)
3. Install and use pre-commit hooks for automated checks
4. Follow our code style guidelines (enforced by Ruff)

For questions or to discuss ideas before sending a PR, hop into the [Discord server](https://discord.gg/PXvBy4NBgv).

## How Positronic differs from LeRobot

If you want to explore ML robotics, prototype policies, or learn the basics, use [LeRobot](https://github.com/huggingface/lerobot). It shines for teaching and fast experiments with imitation/reinforcement learning and public datasets.

If you need to build and operate real applications, use Positronic. Beyond training, it provides the runtime, data tooling, teleoperation, and hardware integrations required to put policies on robots, monitor them, and iterate safely.

- **LeRobot**: Training-centric; quick demos and learning on reference robots and open datasets
- **Positronic**: Lifecycle-centric; immediate-mode middleware ([Pimm](pimm/README.md)), first-class data ops ([Dataset Library](positronic/dataset/README.md)), and hardware-first operations ([Drivers](positronic/drivers), [WebXR](positronic/cfg/webxr.py), [inference](positronic/inference.py))

We use LeRobot's training infrastructure and build on their excellent work. Positronic adds the operational layer that production systems need.

## Roadmap

Our plans evolve with your feedback. Highlights for the next milestones:

- **Delivered**
  - **Policy presets for π₀.₅ and GR00T.** Full support for both architectures.
  - **Remote inference primitives.** Run policies on different machines via unified WebSocket API.
  - **Batch evaluation harness.** `utilities/validate_server.py` for automated checkpoint scoring.
- **Short term**
  - **Richer Positronic Server.** Surface metadata fields, annotation, and filtering flows for rapid triage.
  - **Direct Positronic Dataset integration.** Native adapter for training scripts to stream tensors directly from Positronic datasets.
- **Medium term**
  - **SO101 leader support.** Promote SO101 from follower mode to first-class leader arm.
  - **New operator inputs.** Keyboard and gamepad controllers for quick teleop.
  - **Streaming datasets.** Cloud-ready dataset backend for long-running collection jobs.
  - **Community hardware.** Continue adding camera, gripper, and arm drivers requested by adopters.
- **Long term**
  - **Distributed scheduling.** Cross-machine orchestration on `pimm` for coordinating collectors, trainers, and inference nodes.
  - **Hybrid cloud workflows.** Episode ingestion into object storage with local curation and optional cloud inference.

Let us know what you need on our [Discord server](https://discord.gg/PXvBy4NBgv), drop us a line at hi@positronic.ro or [open a feature request](https://github.com/Positronic-Robotics/positronic/issues/new) at GitHub.
