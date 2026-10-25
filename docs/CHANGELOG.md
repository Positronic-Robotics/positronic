# Changelog

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
