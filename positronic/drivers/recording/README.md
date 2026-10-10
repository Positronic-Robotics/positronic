# Recording and playback

## `DsWriterAgent` (streaming recorder)

Import from `positronic.drivers.recording.ds_writer_agent`.

`DsWriterAgent` records data collection and replay runs. Inference recording belongs to
[`Harness`](../../policy/harness.py), which writes through the same dataset interfaces.
Harness adds `harness.world` and `harness.wall`: first receipt for inputs, emission for commands.
These coordinates align inputs and commands while preserving the original message timestamps.
The agent is a control-loop component (based on our `pimm` library) that turns live inputs into episode recordings using a flexible serializer pipeline. It listens for episode lifecycle commands (start/stop/abort) and, while an episode is open, appends updated inputs with their `pimm.Message.time` timestamps.

Key ideas
- Inputs are registered explicitly through `DsWriterAgent.add_signal(name, serializer=None)`.
- Each `START_EPISODE` names the path its episode records into, and `None` names nowhere. The agent holds no dataset of its own: it opens each named one through the factory it was built with (`DsWriterAgent(LocalDatasetWriter)`), once per name, and closes them all when it stops.
- The agent polls inputs at a configurable rate and appends only on updates.
- Recording is best effort, and this is a deliberate trade rather than an oversight. Each input arrives over a one-slot `pimm` signal where a new value overwrites one still unread, so a recorder that stalls for longer than the gap between two samples loses the older one — commands exactly as much as camera frames or arm state. An episode is what the recorder managed to observe, not a guaranteed-complete log of what happened; treat a missing sample as possible in any analysis that counts them.
- A separate `command` channel controls episode lifecycle.
- Every message coordinate is saved: emission, first delivery, and optional producer timelines.
  Training joins, export, and playback use `harness.world` when present and otherwise `recorded`.
  Replay of data collection recordings requires an explicit `received.world` timeline.
  Pass `--timeline=received.world` to `positronic/replay_record.py` for these recordings.

`Serializer` is a pure function that know how to translate the incoming data into a format that `SignalWriter` can accept:
- A serializer receives the latest value for the input and can return:
  - Transformed value: recorded under the same input name.
  - Dict of suffix -> value: expanded and recorded as `name + suffix` for each
    item (use empty suffix `""` to keep the base name as-is).
  - `None`: the sample is dropped (not recorded).
- Omitting the serializer (or passing `None`) records the value unchanged.

Built‑in serializers (`positronic.dataset.serializers.Serializers`)
- `transform_3d(pose: Transform3D) -> np.ndarray`
  - Returns `[tx, ty, tz, qw, qx, qy, qz]` (shape `(7,)`).
- `robot_state(state: roboarm.State) -> dict | None`
  - Expands to `{'.status': status, '.q': q, '.dq': dq, '.ee_pose': transform_3d(ee)}`; every sample is
    recorded whatever the status.
- `robot_command(command) -> dict`
  - `CartesianPosition(pose)` -> `{'.pose': transform_3d(pose)}`
  - `JointPosition(positions)` -> `{'.joints': positions}`

Lifecycle
- `START_EPISODE`: opens a new episode in the dataset the command names and applies provided static
  metadata (`DsWriterCommand.START(output_path, static_data)`). A command that names no path opens an episode
  that records nowhere, so the `STOP_EPISODE` that ends it is as ordinary as any other.
- `STOP_EPISODE`: finalizes the episode (applies static data then closes).
- `ABORT_EPISODE`: aborts and discards the episode directory.

## `DsPlayerAgent` (dataset player)

Import from `positronic.drivers.recording.ds_player_agent`.

`DsPlayerAgent` replays recorded `Episode` objects back into a live `pimm` world by streaming signal values on demand. It mirrors the lifecycle style of `DsWriterAgent`, making it easy to pipe existing datasets through simulators, robots, or other consumers.

Component layout
- Outputs are dynamically declared via `player.outputs[name]` before playback begins; every declared name must map to a dynamic signal in the episode. Static-only items raise `ValueError`, and missing signals raise `KeyError` so wiring mistakes surface immediately.
- `command` receives control messages. `DsPlayerStartCommand(episode, start_ts=None, end_ts=None, timeline=None)`
  starts playback on the selected timeline, optionally restricting the time window.
  `DsPlayerAbortCommand()` stops immediately without emitting `finished`.
- `finished` emits the originating `DsPlayerStartCommand` once all scheduled samples have been streamed.
- `poll_hz` (default `100 Hz`) governs how frequently the agent checks for new work.

Playback semantics
- `timeline` selects the nanosecond clock used for scheduling and command bounds. When omitted,
  playback uses `harness.world` when present and otherwise `recorded` for legacy data.
  Other timelines require an explicit name. Every requested output must expose the selected
  timeline. Window selection uses the dataset's carry-back semantics.
- Playback anchors the first sample to the world clock when `START` is handled and preserves
  inter-sample spacing. Messages carry this scheduled time as `playback.scheduled`; pimm stamps
  their actual emission and first delivery times.

Typical use cases
- Driving robots or simulators from a stored episode while optionally recording the run again. See [`positronic/replay_record.py`](../../replay_record.py) where `DsPlayerAgent` feeds a Mujoco simulation and simultaneously streams into a `DsWriterAgent` to capture the replay.
- Scrubbing subsets of an episode (via `start_ts`/`end_ts`) for debugging or visualization tools without rewriting the dataset.
- Streaming into analysis pipelines that expect live `pimm` signals; consumers just connect to `player.outputs[...]` as if they were real hardware feeds.
