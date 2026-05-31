# Changelog

## [0.3.0] - 2026-05-31

Inference API rearchitecture. The monolithic `Policy` is split into a stateless `Policy` (the model) and a per-episode `Session`, so one model can serve several robots at once. All scheduling and timing move out of the harness into composable client-side `PolicyWrapper`s, and a command is now a timestamped trajectory the newest prediction overwrites — making latency strategies a client-side choice without any server change. See the new [Connect Your Model](connect-your-model.md) guide.

### Inference API
- **`Policy` → `Policy` + `Session` split**: `Policy.new_session()` returns a callable per-episode `Session`; per-episode state lives in the session so one model serves multiple robots ([#381](https://github.com/Positronic-Robotics/positronic/pull/381))
- **`PolicyWrapper` composition with `|`**: scheduling, error recovery, recording, and codecs compose into one pipeline applied via `.wrap(policy)`; the inference server stays stateless ([#381](https://github.com/Positronic-Robotics/positronic/pull/381))
- **Trajectory-as-command with absolute timestamps**: policies emit timestamped action chunks; the client anchors offsets to its own clock at inference-finish, and a new prediction overwrites the unplayed tail — enabling temporal ensembling and real-time chunking client-side ([#393](https://github.com/Positronic-Robotics/positronic/pull/393))
- **Multi-checkpoint sampling**: `SampledPolicy` + `EpisodeCounter` route each episode to a sub-policy (uniform or balanced) ([#381](https://github.com/Positronic-Robotics/positronic/pull/381))
- **Server-side recording**: composable `Recorder` taps capture both codec boundaries (`raw` wire + `inference`) into one rerun file per episode; `recording_dir` wired through remote policies ([#381](https://github.com/Positronic-Robotics/positronic/pull/381), [#408](https://github.com/Positronic-Robotics/positronic/pull/408))
- **Remote-server authentication** for inference connections ([#394](https://github.com/Positronic-Robotics/positronic/pull/394))
- **Reproducible behavioral golden** for the full inference pipeline ([#400](https://github.com/Positronic-Robotics/positronic/pull/400))
- Old-server compatibility: `RobotStatus` serialized as a str-keyed envelope, decoded from str or bytes ([#381](https://github.com/Positronic-Robotics/positronic/pull/381))

### Docs
- New [Connect Your Model](connect-your-model.md) guide documenting the WebSocket protocol, execution model, and how to plug in a custom server ([#383](https://github.com/Positronic-Robotics/positronic/pull/383), [#385](https://github.com/Positronic-Robotics/positronic/pull/385))
- Rewrote the [Codecs](codecs.md) guide around the training/inference dual path and composability ([#381](https://github.com/Positronic-Robotics/positronic/pull/381))
- Fixed the observation schema and `ee_pose` quaternion order (wxyz) in the inference docs ([#409](https://github.com/Positronic-Robotics/positronic/pull/409))
- Added a demo GIF and decluttered the repo root ([#379](https://github.com/Positronic-Robotics/positronic/pull/379))

### Vendors & Training
- `openpi-server` entrypoint now installs the `openpi` extra ([#380](https://github.com/Positronic-Robotics/positronic/pull/380))
- OpenPI LR schedule spans the full run via `--lr-schedule.decay-steps` ([#391](https://github.com/Positronic-Robotics/positronic/pull/391))
- Dataset sampling for LeRobot conversion and a `droid_spoons` ablation setup ([#389](https://github.com/Positronic-Robotics/positronic/pull/389))

### Dataset
- Lance dataset converter ([#386](https://github.com/Positronic-Robotics/positronic/pull/386))

### PhAIL
- Reorganized PhAIL configs by release version ([#406](https://github.com/Positronic-Robotics/positronic/pull/406))
- Spoons ablation config and USB-C eval objects ([#382](https://github.com/Positronic-Robotics/positronic/pull/382))

### Teleop
- Keyboard `p` finishes and writes the current episode ([#397](https://github.com/Positronic-Robotics/positronic/pull/397))

### Infrastructure
- Nebius serverless workflow for vendor training, dataset conversion, and serving ([#388](https://github.com/Positronic-Robotics/positronic/pull/388))
- Serverless endpoint auto-shutdown and a shared Nebius filesystem caching `uv`/HF/openpi across cold starts ([#390](https://github.com/Positronic-Robotics/positronic/pull/390), [#392](https://github.com/Positronic-Robotics/positronic/pull/392))
- Publish versioned Docker tags only on release ([#407](https://github.com/Positronic-Robotics/positronic/pull/407))
- Docker Compose `pull_policy: always` so the latest image is always pulled ([#384](https://github.com/Positronic-Robotics/positronic/pull/384))
- Bump `pos3` to 0.3.1 ([#411](https://github.com/Positronic-Robotics/positronic/pull/411))

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
