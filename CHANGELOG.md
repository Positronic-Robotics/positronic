# Changelog

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
