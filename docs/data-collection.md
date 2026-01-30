# Data Collection Guide

Positronic provides unified data collection for simulation (MuJoCo) and hardware (Franka, Kinova, SO101, DROID). All demonstrations are recorded as immutable raw datasets that can be projected to any model format using [codecs](codecs.md).

## Quick Start

```bash
uv run positronic-data-collection sim \
    --output_dir=~/datasets/stack_cubes_raw \
    --sound=None \
    --webxr=.iphone \
    --operator_position=.BACK
```

Loads MuJoCo scene, starts DearPyGui visualization, launches WebXR server for phone teleoperation (port 5005), and records to `~/datasets/stack_cubes_raw`. Press `Ctrl+C` to stop.

![Data collection GUI](../positronic/assets/docs/dc_gui.png)

## Phone Teleoperation (WebXR)

### iPhone
```bash
uv run positronic-data-collection sim \
    --output_dir=~/datasets/my_task \
    --webxr=.iphone
```

Install **XR Browser** or **WebXR Viewer** on iPhone. The script prints the URL in console. Open in browser, tap **Enter AR**, grant permissions. Hold phone upright (reticle = virtual controller). Use HUD: **Track** (start/stop positional tracking), **Record** (start/stop episode), **Reset** (abort/reset scene), **Gripper slider** (0-1).

### Android
```bash
uv run positronic-data-collection sim \
    --output_dir=~/datasets/my_task \
    --webxr=.android
```

Use Chrome browser (WebXR built-in). Connect to `https://<host-ip>:5005`, enter AR, control as above.

**Troubleshooting:** If "Enter AR" doesn't appear, try different browser or toggle `--webxr.use_https=True/False`. For jittery tracking, ensure good lighting and hold phone steady when pressing Track. Check firewall allows port 5005 if server is unreachable.

## VR Teleoperation (Meta Quest)

```bash
uv run positronic-data-collection sim \
    --output_dir=~/datasets/my_task \
    --webxr=.oculus
```

Open Oculus Browser, navigate to `https://<host-ip>:5005/`. Browser shows "Dangerous connection" warning (expected with self-signed certificates) – click Advanced → Proceed. Click "Enter AR" and approve permissions.

**Controls:** Right B (start/stop recording), Right A (toggle tracking), Right stick press (abort/reset), Right trigger (gripper).

## Collection Workflow

Press **Track** (or Right A) to enable controller. Press **Record** (or Right B) to start episode. Perform task (e.g., grasp cube, move, place). Press **Record** again to save. Press **Reset** (or Right stick press) to randomize scene and start next episode. Press **Reset** during recording to abort and discard.

**Best practices:** Record calibration runs first and review in Positronic server (`uv run positronic-server --dataset.path=~/datasets/my_task`). Collect 50+ demonstrations for single-task scenarios (minimum 30, multi-task needs 100-500+). Randomize object positions, vary approach angles, keep episodes short (10-30s), demonstrate successful and near-failure cases (not catastrophic failures).

## Hardware Platforms

### Franka Panda
```bash
uv run positronic-data-collection real \
    --output_dir=~/datasets/franka_logistics \
    --webxr=.oculus
```

Requires Franka Panda with FCI, gripper, cameras. Install extras: `uv sync --frozen --extra hardware` (Linux only). Configure network connection and udev rules (see [Drivers](../positronic/drivers/)).

### Other Platforms
- **Kinova Gen3**: Add `--robot_arm=@positronic.cfg.hardware.roboarm.kinova`
- **SO101**: Use `positronic-data-collection so101` (bimanual setup)
- **DROID**: Use `positronic-data-collection droid` (joint-space control)

## Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--output_dir` | Dataset location | `~/datasets/my_task`, `s3://bucket/datasets/task` |
| `--webxr` | Teleoperation | `.iphone`, `.android`, `.oculus`, `None` |
| `--sound` | Audio feedback | `None` (disable), default (enable) |
| `--operator_position` | Camera viewpoint | `.FRONT`, `.BACK`, `.LEFT`, `.RIGHT` |

**S3 support:** Positronic relies on [pos3](https://github.com/Positronic-Robotics/pos3) for S3 integration. Data is cached locally and synced automatically.

**Custom configs:** Add hardware configs in `positronic/cfg/hardware/`, reference with `--robot_arm=@positronic.cfg.hardware.roboarm.my_custom_arm`.

## Reviewing Data

Launch viewer:
```bash
uv run positronic-server \
    --dataset.path=~/datasets/my_task \
    --port=5001
```

Open `http://localhost:5001` to browse episodes, view camera feeds, check timestamps. Delete low-quality episodes manually: `rm -rf ~/datasets/my_task/000000000000/000000000042/`.

![Dataset viewer](../positronic/assets/docs/server_screenshot.png)

## Next Steps

After collection: **Review** data in server → **Curate** by removing failures → **Convert** with codecs ([Training Workflow](training-workflow.md)) → **Train** policy ([Model Selection](model-selection.md)) → **Evaluate** ([Inference Guide](inference.md)).

## See Also

- [Training Workflow](training-workflow.md) – Converting and training
- [Codecs Guide](codecs.md) – Format projection
- [Inference Guide](inference.md) – Policy deployment
- [Dataset Library](../positronic/dataset/README.md) – Raw storage
- [Drivers](../positronic/drivers/) – Hardware integration
