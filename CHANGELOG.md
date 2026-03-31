# Changelog

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
