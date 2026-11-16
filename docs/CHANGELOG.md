# Changelog

## [0.4.6] - 2026-11-12

This release publishes `positronic-platform-client` 0.47.24, and `positronic` requires that version. A submission view lists what it shows for each of its evals, and `positronic eval status` prints that list for a submission of several evals. An `image` endpoint can state the params of its session URL. A session opening of `InferenceClient` and `RoboarenaClient` ends by the connect deadline. The Progress chart of the replay plots the stage and count marks of every channel. The Dream-1 vendor caps the hand speed at 0.3 m/s.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.47.24 |
| `positronic-evals` | 0.2.10 |
| `positronic-eval-vocabulary` | 0.7.5 |
| `positronic-model-server` | 0.3.7 |
| `positronic-wire` | 0.12.3 |

### Platform client
- A submission view carries `evals`: one `EvalView` for each eval of the submission, with its eval, status, scores, artifacts, episode counts and runs. `scores`, `artifacts`, `episodes` and `runs` on the view read the first entry. `list_artifacts` takes `eval`, which picks the eval whose records it lists. `positronic eval status` prints the `evals` list for a submission of several evals ([#1228](https://github.com/Positronic-Robotics/positronic/pull/1228))
- An `image` endpoint states `query`, the params that the platform puts on each session URL of the image, for example `horizon_sec=0.5`. `positronic eval run --policy-query=horizon_sec=0.5` states it beside `--policy-image`, and `ResolvedEndpoint.query` carries it in a resolved plan ([#1229](https://github.com/Positronic-Robotics/positronic/pull/1229))

### Inference
- A caller that passes `connect_deadline` to `InferenceClient` gets one session opening that ends by that deadline: the first dial, each status wait and each retry. `RoboarenaClient` and `RoboarenaPolicy` take `connect_deadline`, with the default `DEFAULT_CONNECT_DEADLINE_S`. `ClientConnection.close` in the wire package takes an optional `timeout` ([#1218](https://github.com/Positronic-Robotics/positronic/pull/1218))
- The FLUX 3 Action pipeline takes `horizon_sec`. A session with the query `horizon_sec=0.5` plays the first 8 commands of each 15 Hz chunk, and a session with no query plays each chunk whole ([#1229](https://github.com/Positronic-Robotics/positronic/pull/1229))
- An ABC-DiT session with real prompts accepts seven more task instructions of ABC's real teleoperation data. A session with sim prompts still refuses them ([#1227](https://github.com/Positronic-Robotics/positronic/pull/1227))
- The Dream-1 chunk limiter moves each hand at most 0.3 m/s (`ChunkLimits.max_ee_speed_m_s`), in place of the 0.6 m/s of the Dream SDK station's defaults ([#1226](https://github.com/Positronic-Robotics/positronic/pull/1226))

### Replay
- The Progress chart of the default replay plots each `<channel>.state` and `<channel>.delivered` signal of any channel, nested names included. A state mark whose values are all stages plots each stage at its place on the stage ladder, so the lines of several channels line up ([#1220](https://github.com/Positronic-Robotics/positronic/pull/1220)) ([#1223](https://github.com/Positronic-Robotics/positronic/pull/1223))

### Development
- A video test compares the frames index, the stream settings, the packet timing and the decoded frames of an early-opened encoder, not the mp4 bytes, which libx264 does not repeat exactly ([#1224](https://github.com/Positronic-Robotics/positronic/pull/1224))

## [0.4.5] - 2026-11-11

This release publishes `positronic-platform-client` 0.47.20, and `positronic` requires that version. `positronic account` gains `redeem`, `terms` and `accept-terms`. An endpoint of a plan can state its own episode cap, policy preset and session limit, and a plan can name the `abc` simulator. The `positronic-server` command is now `positronic-dataset-viewer`. The Franka, ZED and Robotiq drivers report device faults that a console can act on, and the YAM driver can fade its Coulomb friction at rest. The MolmoAct2 pipelines send JPEG frames, and the viewer header counts the outcomes of the episodes.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.47.20 |
| `positronic-evals` | 0.2.8 |
| `positronic-eval-vocabulary` | 0.7.3 |
| `positronic-model-server` | 0.3.6 |
| `positronic-wire` | 0.12.2 |

### Platform client
- A plan can name `abc`, the ABC simulator, as its rig shape. A plan with an unknown shape is refused, and the refusal lists the five shapes ([#1219](https://github.com/Positronic-Robotics/positronic/pull/1219))
- `positronic account redeem --code=<code>` and `PlatformClient.redeem_coupon` add the credits of a coupon to the caller's organization. `--json` prints the answer as JSON. A one-use coupon code may be a command argument, because the same command spends it ([#1207](https://github.com/Positronic-Robotics/positronic/pull/1207))
- `positronic account terms` prints the current platform terms and the acceptance on record. `positronic account accept-terms --version=<v>` accepts a version, and `positronic account register --accept_terms=<v>` accepts it with the key that it mints. `platform_client.terms` holds the models ([#1200](https://github.com/Positronic-Robotics/positronic/pull/1200))
- An endpoint of a plan can state `cap_per_episode_sec` and `policy_preset`, as it states `episodes_per_endpoint`. So one task node runs several policies, each with its own episode window, in one random episode order. `max_cap_per_episode_sec` bounds the cap of each endpoint ([#1199](https://github.com/Positronic-Robotics/positronic/pull/1199))
- An `image` and a `served` endpoint can state `max_concurrent_sessions`, as a `remote` endpoint can. A `nebius_competition` plan refuses the field on every endpoint ([#1167](https://github.com/Positronic-Robotics/positronic/pull/1167)) ([#1182](https://github.com/Positronic-Robotics/positronic/pull/1182))
- A blank eval, board or labeler name is refused with a text that names its kind again, for example "not an eval name: ''" ([#1196](https://github.com/Positronic-Robotics/positronic/pull/1196))
- The client README says that a personal organization buys N credits with the package id `credits-N` ([#1194](https://github.com/Positronic-Robotics/positronic/pull/1194))

### Competition
- The competition guide says why a policy server asks for JPEG frames, and how a server that writes its own handshake or speaks protocol v3 asks for them ([#1212](https://github.com/Positronic-Robotics/positronic/pull/1212))

### Inference
- `--policy.cache_ttl` sets the TTL of the LLM policy's cache points to `5m` (the default) or `1h`, and the policy refuses a provider cache setting ([#1214](https://github.com/Positronic-Robotics/positronic/pull/1214))
- An ABC-DiT session with the query `prompts=real` sends the prompt of the task in ABC's real teleoperation data, in place of the sim prompt. The real data has four of the five tasks, so such a session refuses `hang mug on mug rack` ([#1213](https://github.com/Positronic-Robotics/positronic/pull/1213))
- Every MolmoAct2 DROID pipeline asks the client for JPEG frames at quality 90. A call with three 640x360 cameras sent about 2 MB of raw frames before ([#1212](https://github.com/Positronic-Robotics/positronic/pull/1212))
- `docker/Dockerfile.serve-abc-dit` builds a serving image of ABC-DiT for the yambox rig, with its checkpoint and text encoder inside, so it starts with no network ([#1206](https://github.com/Positronic-Robotics/positronic/pull/1206))
- The Dream-1 vendor paces its chunks as `dreamlabs-client` paces them, and bounds each view to 320 pixels wide before upload. `upload_probe.py` times the frame path of one call ([#1188](https://github.com/Positronic-Robotics/positronic/pull/1188))

### Robots
- The Franka driver holds an ARM `DeviceFault` when the safety controller stops the arm for a brake self-test deadline, a lost safety bus or a joint fault. A console fails the attempt, so the episode runs again ([#1190](https://github.com/Positronic-Robotics/positronic/pull/1190))
- The Franka ready call runs the TD2 self-test when it is due within `SELF_TEST_LEAD_SEC` ([#1189](https://github.com/Positronic-Robotics/positronic/pull/1189))
- A Franka recovery runs the Desk step when Desk holds a recoverable error after libfranka returns `CLEARED`. Only a recover call reboots the control box. A ready call answers with the reason ([#1169](https://github.com/Positronic-Robotics/positronic/pull/1169)) ([#1195](https://github.com/Positronic-Robotics/positronic/pull/1195))
- The lock holds `positronic-franka` 0.12.1, which keeps the playing plan when a new joint target equals its target ([#1179](https://github.com/Positronic-Robotics/positronic/pull/1179))
- A recover call reboots a ZED 2 or ZED 2i camera through the SDK and opens it again ([#1193](https://github.com/Positronic-Robotics/positronic/pull/1193))
- The ZED driver holds a camera error when its images stay near-black for 5 s ([#1192](https://github.com/Positronic-Robotics/positronic/pull/1192))
- A Robotiq move that finds the fingers bound shut holds a GRIPPER `DeviceFault`. A ready or recover call activates the gripper again ([#1197](https://github.com/Positronic-Robotics/positronic/pull/1197))
- `Compensation.coulomb_friction_fade_rad_s` fades the YAM Coulomb friction at low joint speed, and the yambox station uses it, so the arms do not rattle at rest ([#1165](https://github.com/Positronic-Robotics/positronic/pull/1165)) ([#1170](https://github.com/Positronic-Robotics/positronic/pull/1170))
- The default instruction of `.real.yam.bimanual` is the bottles task, 'put the plastic bottles in the bin' ([#1191](https://github.com/Positronic-Robotics/positronic/pull/1191))

### Evals
- `eval_vocabulary.episode.EndedBy` holds the values of `eval.ended_by`: `operator`, `policy` and the new `station`. The sim labelers give no label to a trial that the station ended ([#1177](https://github.com/Positronic-Robotics/positronic/pull/1177))
- The MolmoSpaces eval records the task's reward and its numeric metrics at each step, under `task_metrics` ([#1184](https://github.com/Positronic-Robotics/positronic/pull/1184))
- RoboLab solves pose commands with a port of the FR3 driver's bounded IK ([#1175](https://github.com/Positronic-Robotics/positronic/pull/1175))
- A RoboLab clone whose arm state is not finite ends its own episode as a failure, with `robolab.fault`. The other clones run on ([#1172](https://github.com/Positronic-Robotics/positronic/pull/1172))

### Teleoperation
- The tracker counts the moves of the hand in the axes of the controller, so the way the teleoperator faces does not count. A station sets its own axes with `controller_axes`, and `--controller_axes=None` keeps the axes of the headset ([#1204](https://github.com/Positronic-Robotics/positronic/pull/1204))
- `positronic teleop --cameras` names the cameras of the station. The headset page lists them before Enter AR, a tap shows one in the headset, and Y on the left controller cycles the video panel ([#1198](https://github.com/Positronic-Robotics/positronic/pull/1198))
- The headset page opens its video socket again when no frame arrives for 3 s ([#1174](https://github.com/Positronic-Robotics/positronic/pull/1174))
- The scripted headset holds each press until the page shows it ([#1181](https://github.com/Positronic-Robotics/positronic/pull/1181))

### Datasets
- The dataset viewer package is `positronic.dataset_viewer`, and its command is `positronic-dataset-viewer` in place of `positronic-server`. The viewer cache directory stays at its old default ([#1208](https://github.com/Positronic-Robotics/positronic/pull/1208))
- A `LocalDatasetWriter` opens the video encoders of a new episode before its first frame, at the frame sizes of the last episode ([#1186](https://github.com/Positronic-Robotics/positronic/pull/1186))
- Episode bounds, the static export, the replay recording and the viewer clock skip a signal with no rows ([#1180](https://github.com/Positronic-Robotics/positronic/pull/1180))

### Replay
- The viewer takes an optional `summary_column`, and its page header counts each value of that column with its share of the episodes ([#1201](https://github.com/Positronic-Robotics/positronic/pull/1201))
- `<positronic-replay>` keeps one rerun viewer across episodes ([#1178](https://github.com/Positronic-Robotics/positronic/pull/1178))
- The closed replay screen is inert, so Tab and a screen reader skip its controls ([#1168](https://github.com/Positronic-Robotics/positronic/pull/1168))

### Development
- CI enforces the duplicate-code checks of `utilities/code_smells.toml`, and the code holds no current findings ([#1187](https://github.com/Positronic-Robotics/positronic/pull/1187))
- Each `check-rules` checker prompt starts with a fixed marker line ([#1202](https://github.com/Positronic-Robotics/positronic/pull/1202))
- The next-session teleop test drives the second session until it makes two moves, so a slow runner does not fail it ([#1166](https://github.com/Positronic-Robotics/positronic/pull/1166))

## [0.4.4] - 2026-11-05

This release publishes `positronic-platform-client` 0.47.10, and `positronic` requires that version. `positronic eval run` with no `--eval` runs the eval that the leaderboards tagged `competition.phail.ai` rank, and the competition examples read the `leaderboards` routes. A plan names the simulator it runs in as its rig shape, and a remote endpoint can take a bearer token from a file. A World stops and raises when one of its background processes fails. The serving images no longer trust the leaked eMudhra intermediate CAs. The YAM driver takes the end-effector mass and the Coulomb friction, and `Task.instruction` is plain text that each simulator lists before the reset.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.47.10 |
| `positronic-evals` | 0.2.4 |
| `positronic-eval-vocabulary` | 0.7.1 |
| `positronic-model-server` | 0.3.5 |
| `positronic-wire` | 0.12.2 |

### Platform client
- `PlatformClient.list_leaderboard_evals` reads the slug and the eval of each leaderboard of a tag. It also reads a leaderboard whose other fields this client version refuses. `str()` of a `RankRange` gives `3`, or `1-2` for a range of places ([#1161](https://github.com/Positronic-Robotics/positronic/pull/1161))
- `RigShape` gains `molmospaces` and `robolab`, so a plan names the simulator it runs in. `sim` is removed, and the plan refuses it or any other unknown shape ([#1137](https://github.com/Positronic-Robotics/positronic/pull/1137))
- A `remote` endpoint in a plan file can state `auth_token_file`. `positronic eval run --from-file` reads the bearer token from that file, and the platform presents it to that endpoint only ([#1131](https://github.com/Positronic-Robotics/positronic/pull/1131))
- The plan refuses a bearer token on the `modal_websocket_tls` wire. The READMEs state that the `websocket`, `grpc` and `roboarena` wires have no TLS ([#1136](https://github.com/Positronic-Robotics/positronic/pull/1136))
- The client version moves to 0.47.9 after two changes took the same version ([#1145](https://github.com/Positronic-Robotics/positronic/pull/1145))

### Competition
- `positronic eval run` with no `--eval` runs the eval that the leaderboards tagged `competition.phail.ai` rank. A leaderboard that ranks no eval does not count. `LEADERBOARD_TAG` in `positronic.cli.examples.nebius_competition.leaderboards` names the tag. `standings.py` and `walkthrough.py` read `leaderboards.list` and `leaderboards.get` ([#1161](https://github.com/Positronic-Robotics/positronic/pull/1161))
- CI publishes `positro/gr00t-n17-droid-rtc`, the GR00T N1.7 DROID serving image that starts with the `droid_rtc` preset ([#1142](https://github.com/Positronic-Robotics/positronic/pull/1142))
- The competition guide says that a submission to an eval that the leaderboard does not rank counts for the day ([#1132](https://github.com/Positronic-Robotics/positronic/pull/1132))
- The competition guide names the entry page, where a team can enter and see its submissions ([#936](https://github.com/Positronic-Robotics/positronic/pull/936))

### Inference
- A video policy can take observations at a regular rate while the robot plays its commands. Chunked and RTC policies use one scheduler ([#1139](https://github.com/Positronic-Robotics/positronic/pull/1139))
- A two-arm protocol v3 server can declare `bimanual_joints_action` in its client stack, with an RTC prefix ([#1154](https://github.com/Positronic-Robotics/positronic/pull/1154))
- The LeRobot server serves overlapping sessions, each from its own state, over one copy of the weights ([#1140](https://github.com/Positronic-Robotics/positronic/pull/1140))
- The `dream` vendor serves Dream-1 through DreamAPI to the bimanual YAM over the model-server protocol, with the `dreamlabs-client` package of the `dream` extra ([#1149](https://github.com/Positronic-Robotics/positronic/pull/1149))
- The run command of the Dream-1 server README binds the server to `127.0.0.1`. The README says that the default address is open, and that a server for a robot on another machine needs `AUTH_TOKEN` and a private network ([#1163](https://github.com/Positronic-Robotics/positronic/pull/1163))
- `TransformImages` resizes the selected images of all cameras and frame histories in one bounded thread pool ([#1160](https://github.com/Positronic-Robotics/positronic/pull/1160))
- The serving and training images no longer trust the leaked eMudhra (emSign) intermediate CAs. The main lock holds certifi 2026.10.23 ([#1141](https://github.com/Positronic-Robotics/positronic/pull/1141)) ([#1148](https://github.com/Positronic-Robotics/positronic/pull/1148)) ([#1138](https://github.com/Positronic-Robotics/positronic/pull/1138))

### Runtime
- A World stops when a background process ends, however it ends. A process that fails raises `pimm.BackgroundProcessError` in the parent, and a stop by SIGINT or SIGTERM stays a clean stop ([#1157](https://github.com/Positronic-Robotics/positronic/pull/1157))
- The World keeps its stop flag free of a lock, so a killed process cannot block the others ([#1120](https://github.com/Positronic-Robotics/positronic/pull/1120))

### Robots
- `yam.Robot` takes a `Compensation` in place of `gravity_comp_factor`: the gravity factors, the end-effector mass and the Coulomb friction. Each episode records them per arm ([#1135](https://github.com/Positronic-Robotics/positronic/pull/1135))
- Both yambox YAM arms run the full compensation: the gravity factors, the end-effector mass and the Coulomb friction ([#1162](https://github.com/Positronic-Robotics/positronic/pull/1162))
- The hardware image builds libfranka 0.15.3 or 0.21.3, from the `LIBFRANKA_VERSION` build argument ([#1158](https://github.com/Positronic-Robotics/positronic/pull/1158))
- The Franka DROID camera configs read each camera serial number from the environment ([#1151](https://github.com/Positronic-Robotics/positronic/pull/1151))
- The ZED driver drops a torn frame whose raw buffer starts with zero fill ([#1159](https://github.com/Positronic-Robotics/positronic/pull/1159))

### Evals
- `Task.instruction` is plain text. Each simulator lists the instruction of each task before the reset, and an embodiment whose goal moves during an episode publishes the new goal on its own signal ([#1152](https://github.com/Positronic-Robotics/positronic/pull/1152))
- A recording sync deletes no key under its S3 prefix, so it keeps the episodes that another run uploads there ([#982](https://github.com/Positronic-Robotics/positronic/pull/982))

### LLM policy
- `LLMPolicy` no longer offers the `give_up` tool. `done` is the one tool that ends an episode ([#1156](https://github.com/Positronic-Robotics/positronic/pull/1156))

### Teleoperation
- The `psvr2` preset serves the headset page for PlayStation VR2, with an `immersive-vr` session ([#1129](https://github.com/Positronic-Robotics/positronic/pull/1129))
- The headset page opens its controller socket again after it closes ([#1144](https://github.com/Positronic-Robotics/positronic/pull/1144))
- The WebXR server writes its log to the process log, not to `/tmp/webxr.log` ([#1143](https://github.com/Positronic-Robotics/positronic/pull/1143))
- `positronic.teleop.scripted_headset` does the trials of `positronic teleop` with no person and no headset, for a timing run ([#1125](https://github.com/Positronic-Robotics/positronic/pull/1125))

### Datasets
- `positronic.cfg.video_encoder.nvenc_h264` writes H.264 with the NVENC encoder of an NVIDIA GPU, and `positronic.cfg.dataset_writer.nvenc` selects it. A dataset writer checks its encoder when it is made, so a host that cannot encode fails at start ([#1164](https://github.com/Positronic-Robotics/positronic/pull/1164))

### Replay
- `<positronic-replay>` is a web component in the server's static files. It shows a list of episodes and plays the episode that a visitor opens ([#1134](https://github.com/Positronic-Robotics/positronic/pull/1134))
- `<positronic-replay>` fixes four defects on a host page: the badge and icon styles no longer reach a cell, each copied column name sits over its column, a list keeps the link of another list, and a host page can style the filters and the pressed Details button ([#1155](https://github.com/Positronic-Robotics/positronic/pull/1155))

### Tests
- A remote LLM test bounds the episode length only when inference time is not charged ([#1146](https://github.com/Positronic-Robotics/positronic/pull/1146))

## [0.4.3] - 2026-11-02

This release publishes `positronic-platform-client` 0.47.6, and `positronic` requires that version. The client reads the public leaderboards. `positronic eval status` shows the score of a failed run over the episodes that closed before it failed. A listed submission shows when its run started and ended, and a finished submission can carry a report link. The client refuses a team name over 64 characters or with a control character. A policy can request its next action chunk while the robot executes the current one (real-time chunking), and GR00T is the first model that uses it. A plan can name a Modal Server with sticky sessions as its wire. A person who teleoperates is now an ordinary remote model server. A launcher selects a dataset writer for the video encoding, so code that builds a `Rollout` or a `Harness` passes a writer.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.47.6 |
| `positronic-evals` | 0.2.3 |
| `positronic-eval-vocabulary` | 0.7.1 |
| `positronic-model-server` | 0.3.4 |
| `positronic-wire` | 0.12.2 |

### Platform client
- `PlatformClient` reads the public leaderboards through six keyless routes: the leaderboards of a tag, one leaderboard and its rows, and the charts, episodes, progress and model families of one version. `platform_client.leaderboards` holds their models ([#1108](https://github.com/Positronic-Robotics/positronic/pull/1108))
- A failed run's view has `scores`, the score over the episodes that closed before the run failed, and `positronic eval status` shows it. A failed view without `scores` parses as before ([#1060](https://github.com/Positronic-Robotics/positronic/pull/1060))
- A listed submission has `started_at`, `ended_at` and `estimated_end_at`. Each is `None` until the platform knows it ([#1070](https://github.com/Positronic-Robotics/positronic/pull/1070))
- A finished submission has a `report` link beside `replay`. It is `None` until the platform builds the report ([#1076](https://github.com/Positronic-Robotics/positronic/pull/1076))
- The client README states how the platform replays a scene that a plan pins with `scene_seed`, and how it keys a recorded table by the `clutter` block that the plan states ([#1064](https://github.com/Positronic-Robotics/positronic/pull/1064)) ([#1072](https://github.com/Positronic-Robotics/positronic/pull/1072))

### Competition
- The client removes the spaces at the ends of a team name (`--alias`). It refuses a name with fewer than 1 or more than 64 characters, or with a control character or a text-direction character ([#1112](https://github.com/Positronic-Robotics/positronic/pull/1112))
- The competition guide shows how to fine-tune GR00T N1.7 DROID on MolmoSpaces demonstrations, with a small public dataset ([#1066](https://github.com/Positronic-Robotics/positronic/pull/1066))

### Inference
- Real-time chunking: the client requests the next action chunk while it executes the current plan, and sends the planned commands and the inference latency with the request. GR00T conditions its prediction on them, and the `droid_rtc` preset serves it. An RTC stack needs protocol v2 or later ([#1036](https://github.com/Positronic-Robotics/positronic/pull/1036))
- The `modal_websocket_tls` wire serves a Modal Server with sticky sessions (`@modal.sessioned`, a `.modal.direct` URL). It holds one session per host and per credentials, and a reconnect after a drop reaches the same container. The plan enum gains `Wire.modal_websocket_tls` ([#1106](https://github.com/Positronic-Robotics/positronic/pull/1106))
- A Modal session start pauses 1 s after a failure, then twice the last pause, up to 30 s. `InferenceClient` dial retries use the same schedule, `wire.retry_pauses()` ([#1126](https://github.com/Positronic-Robotics/positronic/pull/1126))
- A parallel codec decodes an RTC reply that comes as a list on the wire ([#1079](https://github.com/Positronic-Robotics/positronic/pull/1079))
- GR00T normalization keeps the planned RTC targets, so a frozen target does not change ([#1082](https://github.com/Positronic-Robotics/positronic/pull/1082))
- With a lead of the whole chunk, `ChunkedSchedule` sends the next call in the round that reads the answer ([#1069](https://github.com/Positronic-Robotics/positronic/pull/1069))
- A policy stack or an answer that the client cannot use raises `positronic.policy.PolicyCommandError`, with the original error as its cause ([#1091](https://github.com/Positronic-Robotics/positronic/pull/1091))
- A WebSocket session that the model server refuses delivers its error to the client before the close, so `RemotePolicy` does not open a new session ([#1061](https://github.com/Positronic-Robotics/positronic/pull/1061))
- An answer reports `elapsed_sec`, the time from submission to completion, and `execution_sec`, the time the call ran. `runtime.has_ready_answers()` tells whether a finished answer is not read yet ([#1115](https://github.com/Positronic-Robotics/positronic/pull/1115))
- `Step.outputs` carries the auxiliary results of a policy step to the processors around it. A codec keeps them, and the wake-up time, when it decodes the commands ([#1116](https://github.com/Positronic-Robotics/positronic/pull/1116))
- The `abc_dit` vendor serves ABC-DiT, the bimanual YAM policy of the ABC project, from its multi-task sim checkpoint ([#1017](https://github.com/Positronic-Robotics/positronic/pull/1017))
- The ABC sim renders from cameras at the ZED-X stations of the ABC-130k dataset by default. `camera_view=d405` keeps ABC's own cameras ([#1104](https://github.com/Positronic-Robotics/positronic/pull/1104))
- The LM IK search returns the iterate with the lowest error, and that error ([#1103](https://github.com/Positronic-Robotics/positronic/pull/1103))
- The bundled YAM model is the model of i2rt at the commit that the `yam` extra pins, with the linear gripper. Every YAM mount is 9.5 mm lower, so the same joints report the same `ee_pose` ([#1124](https://github.com/Positronic-Robotics/positronic/pull/1124))
- The YAM driver takes the joint range from the limits that the chain reports, less a margin ([#1099](https://github.com/Positronic-Robotics/positronic/pull/1099))
- The YAM IK accepts an answer that passes its FK check, whatever the solver reports ([#1095](https://github.com/Positronic-Robotics/positronic/pull/1095))

### LLM policy
- `positronic.vendors.llm.server` serves the LLM policy over the offboard protocol, so it can be a `remote` endpoint. One process serves one model ([#1086](https://github.com/Positronic-Robotics/positronic/pull/1086))
- The tool `move_path` replaces `move_to`. One call carries up to `motion.max_waypoints` (5) waypoints ([#1107](https://github.com/Positronic-Robotics/positronic/pull/1107))
- The default motion bounds are 0.4 m and 90° per move, at 0.15 m/s ([#1097](https://github.com/Positronic-Robotics/positronic/pull/1097))
- The message history is append-only, so the provider prompt cache can match each request ([#1057](https://github.com/Positronic-Robotics/positronic/pull/1057))
- Each episode of an Anthropic eval can take its API key at random from a pool of keys ([#1090](https://github.com/Positronic-Robotics/positronic/pull/1090))
- The results tables name each LLM model by its family and version ([#1085](https://github.com/Positronic-Robotics/positronic/pull/1085))

### Evals
- `--eval.horizon_sec` sets the episode length of every task of a RoboLab run ([#1077](https://github.com/Positronic-Robotics/positronic/pull/1077))
- A RoboLab episode records the scene state that a labeler needs for the stage labels: object poses, gripper contacts and progress per condition ([#1087](https://github.com/Positronic-Robotics/positronic/pull/1087))
- `recorded_try_window` gives the world times of the first and the last sample of the try that an episode directory records. It reads the signal files with pyarrow only ([#1109](https://github.com/Positronic-Robotics/positronic/pull/1109))
- Each ABC eval definition pins `camera_view=zed_x`, so the camera view does not follow the `all_tasks` default ([#1111](https://github.com/Positronic-Robotics/positronic/pull/1111))

### Teleoperation
- `positronic teleop` is a model server, and an eval run connects to it through an ordinary `RemotePolicy` with `cfg.policy.teleop` ([#969](https://github.com/Positronic-Robotics/positronic/pull/969))
- With `AUTH_TOKEN` unset, the teleop service starts on a host and a port and serves with no token ([#1084](https://github.com/Positronic-Robotics/positronic/pull/1084))
- `positronic teleop` shows a desktop notice with a sound while a trial waits for Ready ([#1062](https://github.com/Positronic-Robotics/positronic/pull/1062))
- A MolmoSpaces demonstration collection drops the object image points sensor, so each teleop step runs faster ([#1065](https://github.com/Positronic-Robotics/positronic/pull/1065))

### Datasets
- A launcher selects a dataset writer, and the writer holds the video encoding. `Rollout` carries an unopened writer, and `Harness` opens and closes it in the recording process ([#1080](https://github.com/Positronic-Robotics/positronic/pull/1080))
- `LocalDatasetWriter` locks the dataset directory itself, so no lock file goes to a mirror of the directory ([#1083](https://github.com/Positronic-Robotics/positronic/pull/1083))
- Signal and frames-index parquet files use zstd level 3 and delta and byte-split encodings ([#1088](https://github.com/Positronic-Robotics/positronic/pull/1088))
- `read_spans` and the timing report read zstd-compressed span files (`*.spans.jsonl.zst`) ([#1089](https://github.com/Positronic-Robotics/positronic/pull/1089))
- A Franka episode records `robot_state.ee_wrench`, the external wrench at the end effector. An arm that does not measure it records no new key ([#1071](https://github.com/Positronic-Robotics/positronic/pull/1071))
- A YAM episode records the gravity compensation of each arm's chain in its statics ([#1096](https://github.com/Positronic-Robotics/positronic/pull/1096))
- GR00T arm action labels use the state ahead, and the gripper label uses its command ([#1055](https://github.com/Positronic-Robotics/positronic/pull/1055))
- GR00T fine-tuning keeps the normalization statistics of the base checkpoint ([#1056](https://github.com/Positronic-Robotics/positronic/pull/1056))
- The 3D view of an episode replay opens on a raised side view across the reach ([#1105](https://github.com/Positronic-Robotics/positronic/pull/1105))
- The replay charts of joint commands and joint deltas name each joint ([#1114](https://github.com/Positronic-Robotics/positronic/pull/1114))
- The stats sampler keeps running when a child process exits before its first CPU read ([#1118](https://github.com/Positronic-Robotics/positronic/pull/1118))

### Tests
- The probe child writes its pid file atomically ([#1093](https://github.com/Positronic-Robotics/positronic/pull/1093))
- The teleop camera tests drive the run until the image shows ([#1074](https://github.com/Positronic-Robotics/positronic/pull/1074)) ([#1094](https://github.com/Positronic-Robotics/positronic/pull/1094))

## [0.4.2] - 2026-10-27

This release shows the competition terms when a team enters, and `positronic eval run` with no `--eval` runs the eval that the public leaderboard ranks. It publishes `positronic-platform-client` 0.43.0, and `positronic` requires that version. `robosuite` moves to the new `scenes` extra, so the default install needs no C compiler. Code that imports the MuJoCo scene builder installs `positronic[scenes]`.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.43.0 |
| `positronic-evals` | 0.2.0 |
| `positronic-eval-vocabulary` | 0.6.0 |
| `positronic-model-server` | 0.3.1 |
| `positronic-wire` | 0.12.0 |

### Competition
- `competition-enter` prints the current version of the competition terms, their page, and the command that accepts that version. It reads the public route `GET /v1/competition.terms`, so the preview needs no API key ([#1051](https://github.com/Positronic-Robotics/positronic/pull/1051))
- `positronic eval run` with no `--eval` reads `rankings.list` and runs the eval of the public board, and prints its name. No public board, or public boards over more than one eval, refuse the run and ask for `--eval`. A private run (`--org`) still names its eval ([#1054](https://github.com/Positronic-Robotics/positronic/pull/1054))
- `serverless_eval.py` in the competition examples scores a policy image on a public MolmoSpaces eval with Nebius Serverless, with no local GPU. It deletes the endpoint, the job and the token secret when it ends ([#1031](https://github.com/Positronic-Robotics/positronic/pull/1031))

### Platform client
- An `image` endpoint can state `min_vram_gb`, the least GPU memory that the image needs. The platform gives the run the smallest GPU slice that holds it ([#1044](https://github.com/Positronic-Robotics/positronic/pull/1044))
- A plan or a task can state `scene_seed`. The platform then draws every random part of that task's scene from the seed alone ([#1048](https://github.com/Positronic-Robotics/positronic/pull/1048))

### Inference
- YAM reports and solves its poses at `gripper`, as every Franka arm does since 0.4.1. A policy that states no frame keeps the old YAM point ([#1032](https://github.com/Positronic-Robotics/positronic/pull/1032))
- The openpi subprocess client sends keepalive pings again, and a separate deadline bounds each inference ([#1047](https://github.com/Positronic-Robotics/positronic/pull/1047))
- The first `grip` sample of an episode on a real robot no longer carries the reading from the end of the previous episode ([#1035](https://github.com/Positronic-Robotics/positronic/pull/1035))
- `positro/openpi` builds on a `positro/openpi-base` that CI builds from a pinned commit of the openpi fork ([#1049](https://github.com/Positronic-Robotics/positronic/pull/1049))

### Datasets
- The rows of a LeRobot conversion are picklable, so a conversion with video or with a GR00T recipe runs with DataLoader workers ([#998](https://github.com/Positronic-Robotics/positronic/pull/998))
- Recording starts and orphan sweeps of rerun run one at a time across threads, so a build of recordings in many threads does not deadlock ([#1043](https://github.com/Positronic-Robotics/positronic/pull/1043))
- The episode viewer keeps one material per URDF visual when it tints the robot ([#1042](https://github.com/Positronic-Robotics/positronic/pull/1042))

### Install
- `robosuite` moves from the core dependencies to the `scenes` extra, and the MolmoSpaces install keeps `evdev` out. The install in the competition guide and the local MolmoSpaces eval guide needs no C compiler ([#1045](https://github.com/Positronic-Robotics/positronic/pull/1045))

### Docs
- The competition guide has a Quickstart with the commands of the six steps, and a table that says what goes in each placeholder ([#1053](https://github.com/Positronic-Robotics/positronic/pull/1053))
- The competition guide names only the example images that fit a submission, and lists the five example images where a reader looks for them ([#1041](https://github.com/Positronic-Robotics/positronic/pull/1041)) ([#1052](https://github.com/Positronic-Robotics/positronic/pull/1052))

## [0.4.1] - 2026-10-25

This release moves every Franka arm to the grip point `gripper`. A server that states its frame with `cfg.codecs.franka_hand_frame` must build that entry with 0.4.1 or later: in 0.4.0 the preset names the site `default`, and a 0.4.1 client refuses it. No workspace distribution changes, so the release publishes `positronic` only.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.41.2 |
| `positronic-evals` | 0.2.0 |
| `positronic-eval-vocabulary` | 0.6.0 |
| `positronic-model-server` | 0.3.1 |
| `positronic-wire` | 0.12.0 |

### Inference
- Every Franka arm reports `robot_state.ee_pose` at `gripper` and solves Cartesian commands there: the FR3, the MuJoCo sim, LIBERO, MolmoSpaces and RoboLab. The `default` site is gone ([#1021](https://github.com/Positronic-Robotics/positronic/pull/1021))
- `cfg.codecs.franka_hand_frame` is the Franka Hand point `F_T_EE` measured from `gripper`, and `cfg.codecs.panda_hand_frame` is the hand point of the sim panda. A server trained on poses at one of these points puts it in its stack ([#1021](https://github.com/Positronic-Robotics/positronic/pull/1021))
- A policy that states no `ee_frames` gets each pose at the legacy point of the rig, read from the rig's model, and the harness logs a warning. This fallback is temporary ([#1021](https://github.com/Positronic-Robotics/positronic/pull/1021))
- A version 1 `change_ee_frame` spec is refused as an unsupported version ([#1021](https://github.com/Positronic-Robotics/positronic/pull/1021))
- `FlipGrip(arms=...)` flips the grip of each named arm and publishes `flip_grip` version 2. `FlipGrip()` still publishes version 1 for one arm, and an older client refuses version 2 ([#1034](https://github.com/Positronic-Robotics/positronic/pull/1034))
- The published MolmoAct2 DROID image serves `droid_3cam`, which gives the model both exterior views. The `PIPELINE` build argument or `MOLMOACT2_PIPELINE` at run time selects `droid` for one exterior view ([#1037](https://github.com/Positronic-Robotics/positronic/pull/1037))

### Datasets
- `REAL_ROBOT_TRANSFORM` and `SIM_ROBOT_TRANSFORM` read a recording made before `gripper` and give each pose signal at `gripper`, from the model that the recording carries ([#1021](https://github.com/Positronic-Robotics/positronic/pull/1021))

### Simulation
- `env_server/protocol.py` defines the raw arm observation names `OBS_JOINT_POS`, `OBS_JOINT_VEL`, `OBS_EEF_POS`, `OBS_EEF_QUAT` and `OBS_GRIP`. ABC, LIBERO, MolmoSpaces and RoboLab read them from there ([#1034](https://github.com/Positronic-Robotics/positronic/pull/1034))

### Hardware
- The Franka driver reboots the control box once when Desk is in Programming mode, as for the other faults that only a reboot clears. The `franka` extra requires `positronic-franka` 0.12.0 or later ([#1023](https://github.com/Positronic-Robotics/positronic/pull/1023))

### Docs
- The competition guide says that a Mac or another ARM computer can build and submit a policy image. Only the local image test needs an amd64 Linux computer ([#1038](https://github.com/Positronic-Robotics/positronic/pull/1038))

## [0.4.0] - 2026-10-25

This release changes the platform API contract, and every `positronic` install older than 0.4.0 must upgrade to use the platform. A client from this release and the platform each refuse a version of the other that is too old. This release publishes `positronic-platform-client` 0.41.2, `positronic-evals` 0.2.0, `positronic-eval-vocabulary` 0.6.0 and `positronic-model-server` 0.3.1. `positronic-wire` 0.12.0 does not change, so the release skips it.

| Distribution | Version |
|---|---|
| `positronic-platform-client` | 0.41.2 |
| `positronic-evals` | 0.2.0 |
| `positronic-eval-vocabulary` | 0.6.0 |
| `positronic-model-server` | 0.3.1 |
| `positronic-wire` | 0.12.0 |

### Platform client
- Every request names the client version in the `X-Positronic-Client-Version` header. The platform refuses a client older than 0.40.0, and the refusal names the version to install ([#1024](https://github.com/Positronic-Robotics/positronic/pull/1024))
- The client refuses an answer from a platform that serves a contract older than 0.40.0, or that names no contract version in `X-Positronic-Platform-Version`. Before its first write, the client reads `auth.config` once and checks the version, so it sends no write to such a platform ([#1024](https://github.com/Positronic-Robotics/positronic/pull/1024))
- Each `catalog.tasks` entry lists the rig shapes the task runs on in `embodiments`, in place of the single `embodiment` string. `positronic eval catalog` prints the list. The platform refuses a plan that states its tasks when its `rig_shape` is not one of the `embodiments` of a task ([#1024](https://github.com/Positronic-Robotics/positronic/pull/1024))
- A submitted policy image has two deadlines in place of one shared 1800 s deadline. `IMAGE_FETCH_DEADLINE_S` (1800 s) bounds the VM boot and the image pull, from the start of the VM, and a miss is the platform's fault. `WARMUP_DEADLINE_S` (1800 s) bounds the server's start, from the end of the image pull to the first readiness answer. They replace `PROVISIONING_DEADLINE_S` in `platform_client.policy_container`. A large image no longer shortens the time its server gets to start ([#1028](https://github.com/Positronic-Robotics/positronic/pull/1028))
- `positronic account register` tells an account that already has a key to set `POSITRONIC_PLATFORM_API_KEY` to it, and says that `--rotate` stops the old key ([#989](https://github.com/Positronic-Robotics/positronic/pull/989))
- `platform-register` tells an account that already has a key to set `POSITRONIC_PLATFORM_API_KEY` to it, and says that `platform-register --rotate` stops the old key
- `competition-enter` shows what becomes public about the team and ends with `To join, run: competition-enter --yes`. It does not show the terms. The platform refuses the first submission until the team accepts the terms, and the refusal names `competition-enter --accept-terms=<version>` ([#989](https://github.com/Positronic-Robotics/positronic/pull/989))
- `positronic account register` and `platform-register` print one sign-in line. When the identity provider sends a link that carries the code, the line gives that link, so you type no code. The line also gives the time until the code expires ([#991](https://github.com/Positronic-Robotics/positronic/pull/991))
- When WorkOS signs the user in, the sign-in line of `positronic account register` and `platform-register` tells the user to look in the spam folder for the code mail. The competition guide names the sender, Physical AI Competition (`competition@phail.ai`) ([#1027](https://github.com/Positronic-Robotics/positronic/pull/1027))
- A submitted policy image gets one `1g.24gb` MIG slice of an RTX PRO 6000: 24192 MiB of memory and a quarter of the card's compute ([#996](https://github.com/Positronic-Robotics/positronic/pull/996))
- A plan can state `start_pose: abc_init`, the start pose of each YAM arm in the ABC sim. `abc_init` is a `yam` pose, and `droid_reset` is a `franka` pose ([#1033](https://github.com/Positronic-Robotics/positronic/pull/1033))

### Inference
- A remote model can end its run. The server composes the `end_command` codec with its action codec, and the client stack declares `end_on_field`. When a played element carries `end` (`complete` or `give_up`), the run ends at the next observation ([#1006](https://github.com/Positronic-Robotics/positronic/pull/1006))
- On Linux x86_64, the `molmoact2`, `lerobot` and `lerobot_0_3_3` extras install the CUDA 12.8 build of `torch` 2.7.1, which runs on Blackwell GPUs. This build needs NVIDIA driver 570 or newer. macOS, aarch64 and Windows keep the PyPI build ([#978](https://github.com/Positronic-Robotics/positronic/pull/978))
- The MolmoAct2 server does not start when it finds no CUDA GPU. Before, it served from the CPU, too slowly to act, and an eval scored 0 with no error. `--model.device_map=cpu` still runs it on the CPU ([#1014](https://github.com/Positronic-Robotics/positronic/pull/1014))
- The openpi server sends no keepalive pings to its subprocess. Before, a first inference that compiled for more than about 40 s closed the connection with `1011 keepalive ping timeout`, and the server exited during its warmup ([#1010](https://github.com/Positronic-Robotics/positronic/pull/1010))
- The Galaxea image uses FlashAttention 4 only on compute capability 9 and 10. On other cards, such as the RTX PRO 6000, its vision tower uses its own SDPA path. Before, every call failed there with `Operation creation failed` ([#1011](https://github.com/Positronic-Robotics/positronic/pull/1011))
- A model server stops a session preparation when its client leaves. A preparation that waits for an outside event, such as a person who presses Ready, watches `Model.preparation_cancelled` and raises `PreparationCancelled`. A model that waits for no outside event needs no change ([#1008](https://github.com/Positronic-Robotics/positronic/pull/1008))
- A rig, a recording and a pose policy state the frame of each pose signal in `ee_frames`: `{signal: {site, offset}}`, a site of the robot model and a transform from it. `ee_frames` replaces `control_frame` and `ee_frame`. No pose moves: every rig states site `default` with no offset. `REAL_ROBOT_TRANSFORM` and `SIM_ROBOT_TRANSFORM` read older recordings ([#1020](https://github.com/Positronic-Robotics/positronic/pull/1020))
- At the start of an episode, the harness compares the `ee_frames` of the policy with those of the rig. A policy observes only the pose signals that it states at the site the rig reports. The harness refuses a policy that names another site, and a policy that states frames on a rig that states none, such as Kinova. A GR00T checkpoint that records the old `ee_frame` key is refused ([#1020](https://github.com/Positronic-Robotics/positronic/pull/1020))
- `ChangeEEFrame` takes a site and a transform, and states the frame per key. Wire version 2 carries the site, and version 1 still decodes at `default` ([#1020](https://github.com/Positronic-Robotics/positronic/pull/1020))
- `TransformImages` replaces `RestrictImageSize`. It takes explicit dimensions and a `ResizeMode`: `BOUND` shrinks within the bounds, `PAD` fits and pads to the exact size, and `STRETCH` resizes to the exact size. `StateObservationCodec` and `SelectObservationFields` replace `ObservationCodec`, `PackFields` replaces `PackObservationFields`, and `SplitActionChunk` replaces `UnpackActionChunk`. A remote codec of an older version still decodes. Server metadata does not carry `image_sizes` ([#975](https://github.com/Positronic-Robotics/positronic/pull/975))
- `RemotePolicy.meta()` opens no session and returns only `type: remote`. Each run records the server metadata in the episode, under the same keys. `positronic eval run` calls `wait_for_server()` before the sweep, so a wrong stack or a refused session parameter fails at the first episode ([#1008](https://github.com/Positronic-Robotics/positronic/pull/1008))

### Evaluation
- `positronic eval run --eval=<public eval> --output_dir=<dir>` labels the episodes that the run records and prints their score. It uses the same `positronic_evals` labelers and metric as the platform. Episodes from an earlier run in the same directory do not count ([#999](https://github.com/Positronic-Robotics/positronic/pull/999))
- A run with `--output_dir` on an install without the telemetry extra logs one warning line with the reason, and no traceback ([#999](https://github.com/Positronic-Robotics/positronic/pull/999))
- The static data of an eval episode names only the joint and pose signals that the episode records. Before, an episode with joint commands named `robot_command.pose`, which it does not record ([#1004](https://github.com/Positronic-Robotics/positronic/pull/1004))
- `abc.yam_bimanual` runs the 29 ABC tasks that have a success evaluator, and `abc.yam_bimanual_multitask` the five tasks of the ABC multi-task sim policy. Each task runs 20 trials on the seeds of the ABC sim eval, and a run ranks on the mean of its task success rates. The `abc_label` labeler ends a trial on its recorded verdict ([#1019](https://github.com/Positronic-Robotics/positronic/pull/1019))
- The harness records each instruction that the policy gets as the signal `eval.instruction`, at the first observation and at each change ([#786](https://github.com/Positronic-Robotics/positronic/pull/786))

### Simulation
- `positronic eval run --eval=.sim.amazon_abc.all_tasks` runs a policy against ABC, the MuJoCo sim of the bimanual i2rt YAM: 33 tasks, 29 of them scored. `--eval.task` takes a task name, an alias, a prompt or a list. The ABC server runs in its own interpreter, at a pinned ABC commit ([#786](https://github.com/Positronic-Robotics/positronic/pull/786))
- The MolmoSpaces server reports the end-effector pose relative to the arm base `fr3_link0`, as the real FR3 does. Before, `robot_state.ee_pose` was 0.58 m too high, and a `CartesianPosition` command moved the hand 0.58 m too low ([#972](https://github.com/Positronic-Robotics/positronic/pull/972))
- The MolmoSpaces `benchmarks` eval takes `episode_paths`: episodes from several benchmarks, each as `<suite>/<scene_dataset>/<task_config>/<benchmark>/<episode index>`, run in list order ([#992](https://github.com/Positronic-Robotics/positronic/pull/992))
- The MolmoSpaces eval reports the hand pose at the FR3 tool point, 103.4 mm from the flange, as the real FR3 does. Before, the pose was 51.6 mm further out and turned 45 deg about the approach axis. A Cartesian delta command at `default` works in the MolmoSpaces and RoboLab evals. Before, it failed on the first step ([#994](https://github.com/Positronic-Robotics/positronic/pull/994))
- Two env server launchers on one computer do not get the same port. `serve_subprocess` holds a lock on its port until the launcher exits. Before, two evals could connect to one server, and one of them failed ([#1000](https://github.com/Positronic-Robotics/positronic/pull/1000))
- `DROID_EE_FRAME` is at the flange, where DROID records `cartesian_position`. Before, it was at RoboLab's `eef_frame`, 18.17 mm along the flange axis and turned 90 deg about it. RoboLab's own frame is `ROBOLAB_EE_FRAME` ([#1007](https://github.com/Positronic-Robotics/positronic/pull/1007))
- The bundled sim panda names its flange: a `link8` in `panda.urdf` and a `flange` site in the MuJoCo models. No reported or solved pose changes ([#995](https://github.com/Positronic-Robotics/positronic/pull/995))
- Every bundled Franka model names its grip point `gripper`. The FR3 + 2F-85 site `gripper_grasp_site` is now `gripper` ([#1013](https://github.com/Positronic-Robotics/positronic/pull/1013))

### Hardware
- The Franka driver with `manage_desk=True` starts when Desk or libfranka refuses the start, and streams the reason as an error in place of the arm state. A recover call connects again. It reboots the control box one time when only a reboot clears the failure. `reboot_on_safety_error` is removed from `franka.Robot` and `cfg.hardware.roboarm.franka` ([#976](https://github.com/Positronic-Robotics/positronic/pull/976))
- `DLSIKSolverWithLimits` solves one bounded, damped problem per command, as `positronic-franka` 0.9.0 does. Each joint stays within the model limits and within 0.5 rad of the current joints, and a move near a singular pose is damped. The `franka` extra requires `positronic-franka` 0.9.0 ([#980](https://github.com/Positronic-Robotics/positronic/pull/980))
- The SO-101 and YAM data collection configs `so101cfg` and `yamcfg` are overrides of one `arm_as_gripper` config, which uses the arm driver as the gripper too ([#987](https://github.com/Positronic-Robotics/positronic/pull/987))
- A fault of the Robotiq gripper, the Franka arm or the YAM arm does not end the run. The driver puts a `pimm.SignalError` on the device signal, refuses each move, and keeps the error until a ready or recover call repairs the device ([#993](https://github.com/Positronic-Robotics/positronic/pull/993))
- `Impedance` has a `frame`: the point where the Cartesian stiffness acts, from the receiver's control frame. It defaults to identity, so no behaviour changes. The Franka driver refuses a frame that is not identity ([#1016](https://github.com/Positronic-Robotics/positronic/pull/1016))
- `bundled_yam_model()` carries the i2rt YAM URDF and its visual meshes, with the `default` frame, as the bundled Franka and panda models do. The YAM driver publishes it as its robot metadata, so the 3D viewer and offline IK show and solve the YAM arm ([#1018](https://github.com/Positronic-Robotics/positronic/pull/1018))
- The YAM driver keeps each joint target inside the limits that the i2rt chain reports, less 0.15 rad. Before, a joint 3 target above 3.29 rad moved the arm to the chain limit, 0.1 rad from the point where the chain stops and the motors release the arm ([#1026](https://github.com/Positronic-Robotics/positronic/pull/1026))

### Datasets
- With no layout, a `positronic-server` replay shows the cameras and the 3D view in one grid, with the charts of each arm in a column on its right. A robot with more than one arm shows the charts of each arm as a tab. The Other group holds each signal that no chart plots ([#877](https://github.com/Positronic-Robotics/positronic/pull/877))

### Docs
- The policy image guides state the 24 GB slice. A memory setting that is a fraction of the GPU, such as `XLA_PYTHON_CLIENT_MEM_FRACTION`, is a fraction of the 24 GB ([#996](https://github.com/Positronic-Robotics/positronic/pull/996))
- `docs/local-molmospaces-eval.md` takes you from a clean machine to a local score on `molmo.franka_pick_mini_smoke` and `molmo.franka_pick_mini`: the install, the asset variables, a policy server, the eval command and the dataset viewer ([#999](https://github.com/Positronic-Robotics/positronic/pull/999))
- `docs/local-molmospaces-eval.md` asks for about 80 GB of free disk and lists each part. Before, it asked for 40 GB, which does not hold one run ([#1025](https://github.com/Positronic-Robotics/positronic/pull/1025))
- The GPU descriptions state compute capability 12.0 (sm_120) of the RTX PRO 6000, the PyTorch build to install (CUDA 12.8 or newer), and a command that lists the architectures of the torch in an image ([#1012](https://github.com/Positronic-Robotics/positronic/pull/1012))
- The competition guide lists a C compiler in the requirements. The install builds `evdev` from source ([#999](https://github.com/Positronic-Robotics/positronic/pull/999))
- The competition guide installs `positronic` from the `main` branch, says that the sign-in link expires in 5 minutes, and tells a coding agent to run the register command in the background ([#991](https://github.com/Positronic-Robotics/positronic/pull/991))
- The competition guide registers with `positronic account register`, which saves the key on the computer ([#988](https://github.com/Positronic-Robotics/positronic/pull/988))
- The competition guide links point at `https://competition.phail.ai` ([#986](https://github.com/Positronic-Robotics/positronic/pull/986))

Full list: [v0.3.1...v0.4.0](https://github.com/Positronic-Robotics/positronic/compare/v0.3.1...v0.4.0)

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
