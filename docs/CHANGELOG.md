# Changelog

## [0.3.2] - 2026-10-21

This release publishes `positronic-platform-client` 0.39.2, `positronic-evals` 0.1.1 and `positronic-eval-vocabulary` 0.5.0. The other distributions keep their 0.3.1 versions.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.39.2 |
| `positronic-evals` | 0.1.1 |
| `positronic-eval-vocabulary` | 0.5.0 |
| `positronic-wire` | 0.12.0 |
| `positronic-model-server` | 0.2.3 |

### Platform client
- `positronic account register` tells an account that already has a key to set `POSITRONIC_PLATFORM_API_KEY` to it, and says that `--rotate` stops the old key. `platform-register` says the same, and names `platform-register --rotate` ([#989](https://github.com/Positronic-Robotics/positronic/pull/989))
- `competition-enter` shows what becomes public about the team and ends with `To join, run: competition-enter --yes`. It does not show the terms. The platform refuses the first submission until the team accepts the terms, and the refusal names `competition-enter --accept-terms=<version>` ([#989](https://github.com/Positronic-Robotics/positronic/pull/989))
- `positronic account register` and `platform-register` print one sign-in line. When the identity provider sends a link that carries the code, the line gives that link, so you type no code. The line also gives the time until the code expires ([#991](https://github.com/Positronic-Robotics/positronic/pull/991))
- A submitted policy image gets one `1g.24gb` MIG slice of an RTX PRO 6000: 24192 MiB of memory and a quarter of the card's compute ([#996](https://github.com/Positronic-Robotics/positronic/pull/996))

### Inference
- On Linux x86_64, the `molmoact2`, `lerobot` and `lerobot_0_3_3` extras install the CUDA 12.8 build of `torch` 2.7.1, which runs on Blackwell GPUs. This build needs NVIDIA driver 570 or newer. macOS, aarch64 and Windows keep the PyPI build ([#978](https://github.com/Positronic-Robotics/positronic/pull/978))

### Simulation
- The MolmoSpaces server reports the end-effector pose relative to the arm base `fr3_link0`, as the real FR3 does. Before, `robot_state.ee_pose` was 0.58 m too high, and a `CartesianPosition` command moved the hand 0.58 m too low ([#972](https://github.com/Positronic-Robotics/positronic/pull/972))
- The MolmoSpaces `benchmarks` eval takes `episode_paths`: episodes from several benchmarks, each as `<suite>/<scene_dataset>/<task_config>/<benchmark>/<episode index>`, run in list order ([#992](https://github.com/Positronic-Robotics/positronic/pull/992))

### Hardware
- `DLSIKSolverWithLimits` solves one bounded, damped problem per command, as `positronic-franka` 0.9.0 does. Each joint stays within the model limits and within 0.5 rad of the current joints, and a move near a singular pose is damped. The `franka` extra requires `positronic-franka` 0.9.0 ([#980](https://github.com/Positronic-Robotics/positronic/pull/980))
- The SO-101 and YAM data collection configs `so101cfg` and `yamcfg` are overrides of one `arm_as_gripper` config, which uses the arm driver as the gripper too ([#987](https://github.com/Positronic-Robotics/positronic/pull/987))

### Docs
- The policy image guides state the 24 GB slice. A memory setting that is a fraction of the GPU, such as `XLA_PYTHON_CLIENT_MEM_FRACTION`, is a fraction of the 24 GB ([#996](https://github.com/Positronic-Robotics/positronic/pull/996))
- The competition guide installs `positronic` from the `main` branch, says that the sign-in link expires in 5 minutes, and tells a coding agent to run the register command in the background ([#991](https://github.com/Positronic-Robotics/positronic/pull/991))
- The competition guide registers with `positronic account register`, which saves the key on the computer ([#988](https://github.com/Positronic-Robotics/positronic/pull/988))
- The competition guide links point at `https://competition.phail.ai` ([#986](https://github.com/Positronic-Robotics/positronic/pull/986))

Full list: [v0.3.1...v0.3.2](https://github.com/Positronic-Robotics/positronic/compare/v0.3.1...v0.3.2)

## [0.3.1] - 2026-10-20

This release publishes `positronic-model-server` 0.2.3. The other distributions keep their 0.3.0 versions.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.37.1 |
| `positronic-evals` | 0.1.0 |
| `positronic-eval-vocabulary` | 0.4.0 |
| `positronic-wire` | 0.12.0 |
| `positronic-model-server` | 0.2.3 |

### Platform client
- `competition-enter` reads the key and the platform that `positronic account register` saves, after the environment, as every `positronic` command does. The checkout examples `submit_sample.py`, `walkthrough.py` and `standings.py` read them in the same order ([#979](https://github.com/Positronic-Robotics/positronic/pull/979))
- `competition-enter --accept-terms <version>` accepts the version of the competition terms that the team approved, before the first submission ([#968](https://github.com/Positronic-Robotics/positronic/pull/968))
- `docker/test_policy_image.sh` tests a policy image with one script, and the competition guide puts the image build in its main steps ([#956](https://github.com/Positronic-Robotics/positronic/pull/956))

### Inference
- Chunked inference replans after a robot error by default. A component description with no version asks for the latest version that the client supports, and `positronic-model-server` 0.2.3 writes that by default ([#963](https://github.com/Positronic-Robotics/positronic/pull/963))
- `RestrictImageSize(resample=Resample.AREA)` scales frames with OpenCV `INTER_AREA` ([#948](https://github.com/Positronic-Robotics/positronic/pull/948))
- FLUX 3 Action keeps its text encoder on the CPU until the DiT is packed, so it loads on a 32 GB GPU ([#967](https://github.com/Positronic-Robotics/positronic/pull/967))
- The OpenPI server converts LIBERO units, apart from the command decoding ([#906](https://github.com/Positronic-Robotics/positronic/pull/906))

### Hardware
- `positronic-franka` moves from the `hardware` extra to a new `franka` extra, which requires 0.8.1 or later. A Franka station installs `positronic[hardware,franka]` ([#771](https://github.com/Positronic-Robotics/positronic/pull/771), [#973](https://github.com/Positronic-Robotics/positronic/pull/973))
- `positronic[hardware]` installs on aarch64 Jetson stations, with the ZED SDK 5.4 `pyzed` wheels ([#771](https://github.com/Positronic-Robotics/positronic/pull/771))
- The Franka arm reads AVAILABLE after a fault clear that a caller asked for ([#974](https://github.com/Positronic-Robotics/positronic/pull/974))
- The Desk step of a recover call takes Desk control again when Desk dropped the run's token, and names the login that holds control ([#973](https://github.com/Positronic-Robotics/positronic/pull/973))

### Datasets
- An episode recording keeps every frame of a stored video. `positronic-server` moves its recording cache key to `v3`, so it does not serve a recording from an earlier build ([#977](https://github.com/Positronic-Robotics/positronic/pull/977))

Full list: [v0.3.0...v0.3.1](https://github.com/Positronic-Robotics/positronic/compare/v0.3.0...v0.3.1)

## [0.3.0] - 2026-10-17

This release publishes five more distributions to PyPI beside `positronic`. A `positronic` install pulls each one at the exact version it pins.

| Distribution | Version | Contents |
|---|---|---|
| `positronic-platform-client` | 0.37.1 | The platform wire contract and `platform-register` |
| `positronic-evals` | 0.1.0 | The public eval definitions, labelers and scoring |
| `positronic-eval-vocabulary` | 0.4.0 | The words an attended rollout records |
| `positronic-wire` | 0.12.0 | The client side of the websocket and gRPC inference wires |
| `positronic-model-server` | 0.2.2 | The lightweight model-serving primitives |

### Evaluation
- `positronic eval run` with first-class `Task` and `Eval`, a full suite sweep in one run, and seeded sim trials ([#420](https://github.com/Positronic-Robotics/positronic/pull/420), [#462](https://github.com/Positronic-Robotics/positronic/pull/462), [#428](https://github.com/Positronic-Robotics/positronic/pull/428))
- Remote env-server boundary, with LIBERO, RoboLab and MolmoSpaces behind it ([#450](https://github.com/Positronic-Robotics/positronic/pull/450), [#452](https://github.com/Positronic-Robotics/positronic/pull/452), [#469](https://github.com/Positronic-Robotics/positronic/pull/469), [#504](https://github.com/Positronic-Robotics/positronic/pull/504))
- The edit layer: post-hoc review writes `edits.jsonl` as a view over immutable recordings ([#430](https://github.com/Positronic-Robotics/positronic/pull/430), [#431](https://github.com/Positronic-Robotics/positronic/pull/431), [#899](https://github.com/Positronic-Robotics/positronic/pull/899))
- A browser station console, `positronic-inference web`, with between-episode jog teleop ([#460](https://github.com/Positronic-Robotics/positronic/pull/460), [#463](https://github.com/Positronic-Robotics/positronic/pull/463), [#847](https://github.com/Positronic-Robotics/positronic/pull/847))
- A policy can end its episode, and a plan can state the start pose and sample its trials ([#926](https://github.com/Positronic-Robotics/positronic/pull/926), [#856](https://github.com/Positronic-Robotics/positronic/pull/856), [#868](https://github.com/Positronic-Robotics/positronic/pull/868))

### Inference
- Policies emit timestamped action chunks, and every command channel carries a trajectory ([#393](https://github.com/Positronic-Robotics/positronic/pull/393), [#523](https://github.com/Positronic-Robotics/positronic/pull/523))
- Policies are composable runs of processors ([#652](https://github.com/Positronic-Robotics/positronic/pull/652), [#830](https://github.com/Positronic-Robotics/positronic/pull/830))
- A gRPC wire beside the websocket, a Unix-socket wire, and a `roboarena` wire ([#728](https://github.com/Positronic-Robotics/positronic/pull/728), [#735](https://github.com/Positronic-Robotics/positronic/pull/735), [#787](https://github.com/Positronic-Robotics/positronic/pull/787))
- `RTCSchedule` asks for the next chunk while the current chunk runs, and chunk timing is configurable ([#825](https://github.com/Positronic-Robotics/positronic/pull/825), [#947](https://github.com/Positronic-Robotics/positronic/pull/947))
- Each server serves one checkpoint in the EE frame it declares, and warms at load ([#485](https://github.com/Positronic-Robotics/positronic/pull/485), [#816](https://github.com/Positronic-Robotics/positronic/pull/816), [#754](https://github.com/Positronic-Robotics/positronic/pull/754))
- A direct-API VLM/LLM robot policy ([#781](https://github.com/Positronic-Robotics/positronic/pull/781))

### Vendors and serving images
- MolmoAct2 DROID and MolmoAct2-BimanualYAM ([#453](https://github.com/Positronic-Robotics/positronic/pull/453), [#806](https://github.com/Positronic-Robotics/positronic/pull/806))
- GR00T N1.7 DROID, served in its native environment ([#733](https://github.com/Positronic-Robotics/positronic/pull/733), [#902](https://github.com/Positronic-Robotics/positronic/pull/902))
- Public reference serving images for pi0.5 DROID, GR00T N1.7 DROID, MolmoAct2 DROID, FLUX 3 Action DROID and Cosmos3-Nano DROID, published from CI with the weights inside ([#852](https://github.com/Positronic-Robotics/positronic/pull/852), [#832](https://github.com/Positronic-Robotics/positronic/pull/832), [#835](https://github.com/Positronic-Robotics/positronic/pull/835), [#864](https://github.com/Positronic-Robotics/positronic/pull/864))
- A recipe per model family to build and submit a policy image ([#777](https://github.com/Positronic-Robotics/positronic/pull/777), [#844](https://github.com/Positronic-Robotics/positronic/pull/844))

### Hardware
- i2rt YAM embodiment, with joint-target caps and a park at shutdown ([#501](https://github.com/Positronic-Robotics/positronic/pull/501), [#821](https://github.com/Positronic-Robotics/positronic/pull/821), [#783](https://github.com/Positronic-Robotics/positronic/pull/783))
- Driver-managed Franka Desk: brakes, FCI, `SafetyError` reboot recovery and the Desk self-test ([#471](https://github.com/Positronic-Robotics/positronic/pull/471), [#522](https://github.com/Positronic-Robotics/positronic/pull/522), [#888](https://github.com/Positronic-Robotics/positronic/pull/888))
- Franka joint commands are clamped to the arm's soft limits ([#884](https://github.com/Positronic-Robotics/positronic/pull/884))
- A lost ZED camera reopens, and the SDK reboots a camera it cannot find ([#829](https://github.com/Positronic-Robotics/positronic/pull/829), [#812](https://github.com/Positronic-Robotics/positronic/pull/812))
- The gripper convention is 0 = open, 1 = closed ([#466](https://github.com/Positronic-Robotics/positronic/pull/466))

### Platform client
- `positronic eval run --policy-image` submits a policy image to the platform, and `positronic eval status`, `list`, `cancel` and `catalog` read back what it did ([#601](https://github.com/Positronic-Robotics/positronic/pull/601), [#712](https://github.com/Positronic-Robotics/positronic/pull/712))
- `platform-register` signs in through the device flow that the platform's `auth.config` names, and waits 90 s on a register with no timeout ([#706](https://github.com/Positronic-Robotics/positronic/pull/706), [#959](https://github.com/Positronic-Robotics/positronic/pull/959), [#964](https://github.com/Positronic-Robotics/positronic/pull/964))
- The competition-enter command, and a guide for participants of the Physical AI Competition ([#919](https://github.com/Positronic-Robotics/positronic/pull/919), [#890](https://github.com/Positronic-Robotics/positronic/pull/890))
- `positronic account register` and `positronic account credits`, with billing organization discovery, prepaid credits and rig contracts ([#870](https://github.com/Positronic-Robotics/positronic/pull/870), [#871](https://github.com/Positronic-Robotics/positronic/pull/871), [#886](https://github.com/Positronic-Robotics/positronic/pull/886))

### Core
- `World` owns time through a virtual clock, and pimm has request/reply between control systems ([#413](https://github.com/Positronic-Robotics/positronic/pull/413), [#636](https://github.com/Positronic-Robotics/positronic/pull/636))
- Telemetry sidecars record wall-clock spans and machine stats beside the dataset ([#531](https://github.com/Positronic-Robotics/positronic/pull/531), [#811](https://github.com/Positronic-Robotics/positronic/pull/811))
- Immutable `Time` with named coordinate operations, and symmetric named timelines in datasets ([#838](https://github.com/Positronic-Robotics/positronic/pull/838), [#841](https://github.com/Positronic-Robotics/positronic/pull/841))
- The viewer renders under a prefix and exports static files one episode at a time ([#704](https://github.com/Positronic-Robotics/positronic/pull/704), [#932](https://github.com/Positronic-Robotics/positronic/pull/932))

Full list: [v0.2.1...v0.3.0](https://github.com/Positronic-Robotics/positronic/compare/v0.2.1...v0.3.0)

## [0.2.1] - 2026-04-01

### Fixes
- Fix PyPI installability: `openpi-client` moved from core dependency to optional `[openpi]` extra — resolves numpy version conflict with `rerun-sdk==0.30.0` that made `pip install positronic` fall back to 0.1.1
- DreamZero vendor now uses `positronic.utils.serialization` instead of `openpi_client.msgpack_numpy` (wire-compatible, same format)
- Default server port changed from 5000 to 8400

## [0.2.0] - 2026-03-31

Full pipeline and infrastructure for the [PhAIL](https://phail.ai) launch: real-robot evaluation of VLA models on commercial picking tasks with production metrics (UPH, completion rate, availability).

### PhAIL
- Evaluation harness with RUN/STOP/HOME directives and balanced multi-policy sampling ([#366](https://github.com/Positronic-Robotics/positronic/pull/366), [#370](https://github.com/Positronic-Robotics/positronic/pull/370), [#373](https://github.com/Positronic-Robotics/positronic/pull/373))
- Production visualization server with leaderboard, baselines, and per-object metric aggregation ([#338](https://github.com/Positronic-Robotics/positronic/pull/338), [#340](https://github.com/Positronic-Robotics/positronic/pull/340), [#376](https://github.com/Positronic-Robotics/positronic/pull/376))
- Dataset release pipeline with versioned public S3 layout and verification ([#376](https://github.com/Positronic-Robotics/positronic/pull/376))
- Unified eval configs with UPH and MTBF metrics ([#286](https://github.com/Positronic-Robotics/positronic/pull/286))
- Per-machine Docker Compose configs for multi-robot deployment ([#344](https://github.com/Positronic-Robotics/positronic/pull/344), [#356](https://github.com/Positronic-Robotics/positronic/pull/356))

### Vendors
- **LeRobot 0.4.x** (SmolVLA, ACT, Diffusion) – full training and inference ([#328](https://github.com/Positronic-Robotics/positronic/pull/328))
- **DreamZero** (NVIDIA 14B World Action Model) – inference, LoRA fine-tuning, wan2.1/wan2.2 backbones ([#326](https://github.com/Positronic-Robotics/positronic/pull/326), [#333](https://github.com/Positronic-Robotics/positronic/pull/333), [#369](https://github.com/Positronic-Robotics/positronic/pull/369)). *Work in progress.*
- GR00T N1.6 update with 6D rotation representation ([#266](https://github.com/Positronic-Robotics/positronic/pull/266), [#268](https://github.com/Positronic-Robotics/positronic/pull/268))
- Refactored vendor-specific code into `positronic/vendors/` modules ([#278](https://github.com/Positronic-Robotics/positronic/pull/278), [#288](https://github.com/Positronic-Robotics/positronic/pull/288))
- Extracted `VendorServer` base class with startup warmup and dynamic checkpoint loading ([#259](https://github.com/Positronic-Robotics/positronic/pull/259), [#330](https://github.com/Positronic-Robotics/positronic/pull/330), [#352](https://github.com/Positronic-Robotics/positronic/pull/352))

### Codecs & Inference
- Composable `Codec` with `|` and `&` operators ([#307](https://github.com/Positronic-Robotics/positronic/pull/307), [#312](https://github.com/Positronic-Robotics/positronic/pull/312))
- Client-side codecs decoupled from inference servers ([#313](https://github.com/Positronic-Robotics/positronic/pull/313))
- Trajectory-based codec variants for data conversion ([#296](https://github.com/Positronic-Robotics/positronic/pull/296))
- Offboard inference server and remote policy ([#246](https://github.com/Positronic-Robotics/positronic/pull/246), [#265](https://github.com/Positronic-Robotics/positronic/pull/265))
- Status-based WebSocket protocol for model loading ([#283](https://github.com/Positronic-Robotics/positronic/pull/283))
- Policy refactored to return action chunks with delegated buffering ([#254](https://github.com/Positronic-Robotics/positronic/pull/254))
- Joint-space targets from recorded EE targets via IK ([#347](https://github.com/Positronic-Robotics/positronic/pull/347))

### Dataset
- Remote dataset server and client for HTTP-based access ([#269](https://github.com/Positronic-Robotics/positronic/pull/269))
- Public dataset access via `positronic-public` S3 bucket ([#270](https://github.com/Positronic-Robotics/positronic/pull/270))
- Dataset quality signals and episode filtering ([#322](https://github.com/Positronic-Robotics/positronic/pull/322))
- `diff`, `norm`, and scalar aggregators for signal transforms ([#323](https://github.com/Positronic-Robotics/positronic/pull/323))
- Generalized migration tool to work with any `Dataset` source ([#376](https://github.com/Positronic-Robotics/positronic/pull/376))
- Lazy evaluation for episode transforms ([#272](https://github.com/Positronic-Robotics/positronic/pull/272))
- Dataset configs refactored into `cfg/ds/` with public/internal separation ([#277](https://github.com/Positronic-Robotics/positronic/pull/277))
- Mark unfinished episodes and ignore them on reading ([#250](https://github.com/Positronic-Robotics/positronic/pull/250))

### Visualization
- 3D trajectory visualization with URDF robot model in Rerun viewer ([#318](https://github.com/Positronic-Robotics/positronic/pull/318), [#362](https://github.com/Positronic-Robotics/positronic/pull/362))
- Filtered episode navigation and URL-based time seek ([#325](https://github.com/Positronic-Robotics/positronic/pull/325), [#332](https://github.com/Positronic-Robotics/positronic/pull/332))
- Optimized RRD generation: AssetVideo, send_columns, fixed O(N²) trajectory trail ([#320](https://github.com/Positronic-Robotics/positronic/pull/320), [#321](https://github.com/Positronic-Robotics/positronic/pull/321))
- `RecordingCodec` for rerun-based inference introspection ([#310](https://github.com/Positronic-Robotics/positronic/pull/310))

### Hardware
- Make Franka `Robot` picklable by deferring connection ([#360](https://github.com/Positronic-Robotics/positronic/pull/360))
- Explicit `Recover` command for robot error recovery ([#308](https://github.com/Positronic-Robotics/positronic/pull/308))
- Franka driver upgrade with `ee_wrench` capabilities

### Infrastructure
- Extracted `pos3` S3 library to a separate package
- `lerobot` made an optional dependency ([#324](https://github.com/Positronic-Robotics/positronic/pull/324))
- Packaging CI check for missing data files ([#358](https://github.com/Positronic-Robotics/positronic/pull/358))
- Parameterized Docker image tags for parallel branch workflows ([#329](https://github.com/Positronic-Robotics/positronic/pull/329))

## [0.1.1] - 2025-10-07

This release is focused on addressing bugs and subtle inaccuracies of v0.1.0. With it, the end-to-end workflow as described in [README.md](README.md) is working.

### Added
- Observer system to MujocoSim for computing simulation metrics ([#158](https://github.com/Positronic-Robotics/positronic/pull/158))
- `DsPlayerAgent` to replay datasets with `replay_record.py` script ([#155](https://github.com/Positronic-Robotics/positronic/pull/155))
- Movement sensitivity parameter for controller input ([#152](https://github.com/Positronic-Robotics/positronic/pull/152))
- Discord webhook integration for merged PR announcements ([#153](https://github.com/Positronic-Robotics/positronic/pull/153))

### Changed
- `InferenceCommand` class to manage inference lifecycle ([#149](https://github.com/Positronic-Robotics/positronic/pull/149))
- Split `positronic.dataset.transforms` into several submodules for better organization ([#157](https://github.com/Positronic-Robotics/positronic/pull/157))
- Refactored `RelativeTargetPositionAction` to use `robot_state.ee_pose` for position and quaternion inputs, removing the `RelativeRobotPositionAction` class ([#166](https://github.com/Positronic-Robotics/positronic/pull/166))
- Updated dataset transforms and policy wiring to align with LeRobot updates ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- Encode state in inference consistently to data collection for E2E workflow ([#166](https://github.com/Positronic-Robotics/positronic/pull/166))
- Enabled true streaming of episode RRD data over FastAPI ([#160](https://github.com/Positronic-Robotics/positronic/pull/160))
- Enhanced WebXR iPhone HUD button behavior with toggle states ([#151](https://github.com/Positronic-Robotics/positronic/pull/151))
- Bundled rerun viewer assets locally and updated episode template to self-host the iframe ([#150](https://github.com/Positronic-Robotics/positronic/pull/150))
- Improved GUI camera initialization and WebXR host discovery ([#148](https://github.com/Positronic-Robotics/positronic/pull/148))
- Formalized contribution guidelines and updated workflow checks ([#156](https://github.com/Positronic-Robotics/positronic/pull/156))
- Updated action configuration approach ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))

### Fixed
- Fixed bug where simulator data reported the same values across episode ([#162](https://github.com/Positronic-Robotics/positronic/pull/162))
- Home directory resolution (~) support to LocalDataset and LocalDatasetWriter ([#154](https://github.com/Positronic-Robotics/positronic/pull/154))
- Fixed inference reset issues in MujocoSim ([#158](https://github.com/Positronic-Robotics/positronic/pull/158))
- Fixed wrong feature name in observation encoder ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- Fixed wrong image resizing call ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- Fixed bug in ActionDecoder implementations ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- Fixed case where signals are empty in Episode duration property ([#165](https://github.com/Positronic-Robotics/positronic/pull/165))
- Fixed formatting and lint errors ([#159](https://github.com/Positronic-Robotics/positronic/pull/159))
- Fixed bug when mujoco state was written at the end of episode instead of at the beginning ([#155](https://github.com/Positronic-Robotics/positronic/pull/155))
- Reset method to PI0RemotePolicy class ([#149](https://github.com/Positronic-Robotics/positronic/pull/149))

### Improved
- Raised clear error when dataset path is wrong ([#163](https://github.com/Positronic-Robotics/positronic/pull/163))
- Removed excessive FPS logging in MujocoCamera ([#167](https://github.com/Positronic-Robotics/positronic/pull/167), [#165](https://github.com/Positronic-Robotics/positronic/pull/165))
- Inference script now reports meta information and success metrics for simulation tasks ([#161](https://github.com/Positronic-Robotics/positronic/pull/161))
- Added utility for exporting transformed datasets to disk ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- Introduced `KeyFuncEpisodeTransform` to simplify creating EpisodeTransforms ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- EpisodeWriter now aborts when context is exited with exception ([#164](https://github.com/Positronic-Robotics/positronic/pull/164))
- Increased queue size of Inference.command receiver to avoid drops of reset command ([#158](https://github.com/Positronic-Robotics/positronic/pull/158))
- Device detection function for inference ([#149](https://github.com/Positronic-Robotics/positronic/pull/149))

## [0.1.0] - 2025-09-25

Initial release.
