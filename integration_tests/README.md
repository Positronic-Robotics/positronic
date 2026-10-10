# Integration tests

These scripts exercise complete runs through public entry points. Their model servers and renderers
are explicit prerequisites; they do not run as part of the normal pytest suite.

## ACT cube stacking

`act_stack.py` runs the public ACT checkpoint `050000` against the native MuJoCo stacking scene.
Each seed (4 and 8) runs in its own eval process for 15 simulated seconds. Simulation pauses during
inference, so network and model latency do not change the trial's time budget.
This covers the ACT stacking path; other policies and error handling need their own tests.

The default checks:

- **Task success:** the green cube touches the red cube, sits 15–25 mm above its center, and touches
  neither finger, continuously for at least 0.5 seconds. The check reconstructs every recorded physics
  sample; it also requires a complete episode with the requested seed and checkpoint. Robot pose,
  joints and gripper observations must each contain one sample at every physics step.
- **Camera recordings:** both camera videos decode completely and their frame counts match their
  timestamp indexes. Frame timestamps must be nonempty and ordered.
- **Exact behavior:** commands, robot pose, joints, gripper state, both cube poses, support state and
  their episode-relative timestamps match the committed reference arrays. The comparison reports the
  first differing field, sample time and value.

Traces use `harness.world`: command emission and input receipt time. Recordings without it use
`received.world` or legacy `recorded`. Comparisons exclude wall time, which varies between runs.

The references are tied to the environment in `fixtures/act_stack/provenance.json`. Exact matching
across other GPUs or rendering environments is unverified. `--success_only=True` explicitly runs the task
success check on those setups; it never substitutes for a failed exact comparison.

### Start the server

Use the predefined Docker Compose service from the checkout whose server code you want to exercise:

```bash
CACHE_ROOT=/home/vertix docker --context notebook compose -f docker/docker-compose.yml \
  run --rm --no-deps --name act-integration-server -p 127.0.0.1:18024:8000 \
  lerobot-0_3_3-server demo --model.checkpoint=050000
```

`CACHE_ROOT` is the remote user's home directory. `IMAGE_TAG` selects a built image; use a matching
image when changing server code. See `docker/CONTEXTS.md` and `docker/Makefile`.

In another terminal, forward the endpoint:

```bash
ssh -N -L 18024:127.0.0.1:18024 notebook
```

The server is ready when `curl --fail -X POST http://localhost:18024/api/v1/keepalive` answers.
The eval client and MuJoCo run locally, with a working renderer; model prediction runs on the server.
A local server can instead be started with the same Compose service and `--service-ports`.

### Run and inspect

From the repository root:

```bash
uv run --locked python integration_tests/act_stack.py run \
  --host=localhost --port=18024 --output_dir=/tmp/act-integration-run
```

The output directory must be new. Recordings and one eval log per seed remain there on success or
failure. A failed check, model error, or eval process exceeding `--wall_timeout` (120 seconds by
default) exits nonzero. The two episodes cover 30 simulated seconds; wall time depends on the model
and renderer. `--seeds='[4]'` selects one case. The CLI uses Configuronic, like the eval command.

Check saved recordings without running the model again:

```bash
uv run --locked python integration_tests/act_stack.py check --output_dir=/tmp/act-integration-run
```

Use `--success_only=True` with `run` or `check` to check task completion without the hardware-specific
reference. Stop the temporary server and SSH tunnel when finished.

### Capture an intentional reference change

Run successful episodes with `--success_only=True`, then capture into a new directory:

```bash
uv run --locked python integration_tests/act_stack.py capture \
  --output_dir=/tmp/act-integration-run --reference_dir=/tmp/new-act-reference
```

Capture requires successful, complete recordings and refuses to overwrite an existing reference
directory. It publishes the directory after every selected seed validates and writes successfully;
a failed capture can be retried with the same destination. Review the trajectories and repeatability,
then replace the two fixture files and update `provenance.json` with the code revision, checkpoint
and environments that produced them. The normal
run never updates its own expectations. Fixtures use NumPy's `.npz` format with lossless LZMA ZIP
compression to stay below the repository's file-size limit.

The CPU checks for the checker itself can be run separately:

```bash
uv run --locked pytest integration_tests/tests -n0
```

## Background camera recording

The process-level test in `positronic/tests/test_inference.py` runs a synthetic camera through
pimm shared memory and records two consecutive episodes. It checks that policy image snapshots
remain stable while the camera reuses its buffer. Each completed episode is opened immediately;
decoded frame pixels must match their producer frame IDs, and every indexed frame must decode.
The test also checks emission and receipt timestamps and episode metadata.

This test runs in the normal pytest suite and needs no physical camera or model server:

```bash
uv run --locked pytest positronic/tests/test_inference.py::test_the_web_console_records_each_episode_with_its_instruction_and_verdict -n0
```
