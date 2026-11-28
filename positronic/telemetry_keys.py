"""Canonical span names and attribute keys of the eval telemetry sidecars.

These literals are the producer↔reducer contract: a producer opens a span (or stamps an attribute) by
them and the offline reduce (``positronic.cli.eval.timing_report``) matches on the same ones. Defining
them once makes a rename a single-site change the type checker propagates, instead of a string literal
duplicated across the harness, the eval CLI, the sim adapters and the report.

They live here rather than in ``positronic.telemetry`` because **a name belongs to whoever writes the
bytes it names**. These are written by eval-domain code THROUGH the mechanism, which passes them opaquely
and never matches on them; ``positronic.telemetry`` owns the names of what it writes itself — the
machine-load sample's fields, the sidecar file suffixes, the telemetry subdirectory. That is what keeps
the mechanism domain-blind: it could not hold ``SPAN_EPISODE`` without knowing what an episode is.

The module imports only the stdlib-only env-server writer, so any producer can reach it.
"""

# ``env.step``/``env.reset`` are owned by the stdlib-only env-server writer — the isolated env interpreter
# cannot import positronic — and re-exported here for the main process's producers and reduce.
from positronic.simulator.env_server.telemetry import SPAN_ENV_RESET as SPAN_ENV_RESET
from positronic.simulator.env_server.telemetry import SPAN_ENV_STEP as SPAN_ENV_STEP

SPAN_EVAL_PASS = 'eval.pass'
SPAN_EPISODE = 'episode'
SPAN_RESET = 'reset'
SPAN_MATERIALIZE = 'materialize'
SPAN_POLICY_INFER = 'policy.infer'
SPAN_RECORD_IO = 'record.io'

# Work submitted to a worker; its parent is the processor call that submitted it.
SPAN_POLICY_SUBMIT = 'policy.submit'
SPAN_POLICY_ENCODE = 'policy.encode'
SPAN_POLICY_PREPARE = 'policy.prepare'
SPAN_WIRE_SEND = 'wire.send'
SPAN_WIRE_RECV = 'wire.recv'

ATTR_EPISODE_INDEX = 'episode.index'
ATTR_EPISODE_STEPS = 'episode.steps'
ATTR_EPISODE_VIRTUAL_S = 'episode.virtual_s'
ATTR_EPISODE_PARTIAL = 'episode.partial'
# On `episode`: the count of harness steps that start N ms or more after their due time (`steps_late_ge_<N>ms`),
# the count of the episode's `policy.encode` spans, the count of those that take more than N ms
# (`encodes_gt_<N>ms`), and the sum of their durations.
ATTR_EPISODE_STEPS_LATE_GE_1MS = 'episode.steps_late_ge_1ms'
ATTR_EPISODE_STEPS_LATE_GE_2MS = 'episode.steps_late_ge_2ms'
ATTR_EPISODE_STEPS_LATE_GE_5MS = 'episode.steps_late_ge_5ms'
ATTR_EPISODE_STEPS_LATE_GE_10MS = 'episode.steps_late_ge_10ms'
ATTR_EPISODE_STEPS_LATE_GE_20MS = 'episode.steps_late_ge_20ms'
ATTR_EPISODE_STEPS_LATE_GE_50MS = 'episode.steps_late_ge_50ms'
ATTR_EPISODE_ENCODES = 'episode.encodes'
ATTR_EPISODE_ENCODES_GT_5MS = 'episode.encodes_gt_5ms'
ATTR_EPISODE_ENCODES_GT_10MS = 'episode.encodes_gt_10ms'
ATTR_EPISODE_ENCODES_GT_15MS = 'episode.encodes_gt_15ms'
ATTR_EPISODE_ENCODES_GT_30MS = 'episode.encodes_gt_30ms'
ATTR_EPISODE_ENCODES_GT_60MS = 'episode.encodes_gt_60ms'
ATTR_EPISODE_ENCODES_SUM_MS = 'episode.encodes_sum_ms'
# On `episode`: the camera frame reads of the steps, the count of those whose frame waits N ms or more after its
# hand-off, and the sum of the waits. Then the camera frames whose driver stamps a capture time, the count of those
# whose driver takes N ms or more from capture to hand-off, and the sum of the driver times.
ATTR_EPISODE_FRAME_READS = 'episode.frame_reads'
ATTR_EPISODE_FRAME_READS_WAIT_GE_5MS = 'episode.frame_reads_wait_ge_5ms'
ATTR_EPISODE_FRAME_READS_WAIT_GE_10MS = 'episode.frame_reads_wait_ge_10ms'
ATTR_EPISODE_FRAME_READS_WAIT_GE_20MS = 'episode.frame_reads_wait_ge_20ms'
ATTR_EPISODE_FRAME_READS_WAIT_GE_50MS = 'episode.frame_reads_wait_ge_50ms'
ATTR_EPISODE_FRAME_READS_WAIT_GE_100MS = 'episode.frame_reads_wait_ge_100ms'
ATTR_EPISODE_FRAME_READS_WAIT_SUM_MS = 'episode.frame_reads_wait_sum_ms'
ATTR_EPISODE_FRAMES_DRIVER_TIMED = 'episode.frames_driver_timed'
ATTR_EPISODE_FRAMES_DRIVER_GE_10MS = 'episode.frames_driver_ge_10ms'
ATTR_EPISODE_FRAMES_DRIVER_GE_20MS = 'episode.frames_driver_ge_20ms'
ATTR_EPISODE_FRAMES_DRIVER_GE_50MS = 'episode.frames_driver_ge_50ms'
ATTR_EPISODE_FRAMES_DRIVER_GE_100MS = 'episode.frames_driver_ge_100ms'
ATTR_EPISODE_FRAMES_DRIVER_GE_200MS = 'episode.frames_driver_ge_200ms'
ATTR_EPISODE_FRAMES_DRIVER_SUM_MS = 'episode.frames_driver_sum_ms'
ATTR_PASS_FAILED = 'pass.failed'

# What the server reported, stamped on `policy.infer` under this prefix — `served.infer_ms`, its sibling
# durations and the instants `served.received_ns` and `served.sent_ns`, straight from the answer's own `timing` block.
ATTR_SERVED_PREFIX = 'served.'

# On `policy.submit`: the age in milliseconds of the newest frame of each camera that the harness reads, the oldest
# of them, when the call starts. The history frames of a temporal stack do not count.
ATTR_OBS_AGE_MS = 'obs.age_ms'

# Which codec a `policy.encode` span timed, and how many bytes the observation took on the wire.
ATTR_CODEC = 'codec'
ATTR_WIRE_BYTES = 'wire.bytes'
# On `policy.infer`, in milliseconds: from the send to the server's receipt, from the server's send to the receipt, and
# their sum. The first two hold the offset between the clocks of the client and the server; the sum does not.
ATTR_WIRE_UPLINK_MS = 'wire.uplink_ms'
ATTR_WIRE_DOWNLINK_MS = 'wire.downlink_ms'
ATTR_WIRE_NETWORK_MS = 'wire.network_ms'
# On `policy.infer`: whether the client sent the call with no earlier call in flight (docs/telemetry.md).
ATTR_WIRE_SENT_ALONE = 'wire.sent_alone'
# On `wire.send`: whether the send is the first on its connection. The first send also carries the URL that the
# client dialed and the milliseconds that the dial took.
ATTR_WIRE_NEW_CONNECTION = 'wire.new_connection'
ATTR_WIRE_ENDPOINT = 'wire.endpoint'
ATTR_WIRE_CONNECT_MS = 'wire.connect_ms'

# The harness process's sidecar name — the discriminator between client-side spans (episode, client env.step)
# and an env server's own file, which reduces rely on.
HARNESS_PROCESS = 'harness'

# One harness step: its durations in milliseconds, and the highest generation a collection reached. docs/telemetry.md
# defines each value.
SPAN_HARNESS_STEP = 'harness.step'
ATTR_STEP_LATE_MS = 'step.late_ms'
ATTR_STEP_OBSERVE_MS = 'step.observe_ms'
ATTR_STEP_POLICY_MS = 'step.policy_ms'
ATTR_STEP_EMIT_MS = 'step.emit_ms'
ATTR_STEP_GC_WAIT_MS = 'step.gc_wait_ms'
ATTR_STEP_GC_MS = 'step.gc_ms'
ATTR_STEP_GC_GENERATION = 'step.gc_generation'
ATTR_STEP_RECORD_WAIT_MS = 'step.record_wait_ms'
# The signal's name follows each prefix.
ATTR_STEP_READ_MS_PREFIX = 'step.read_ms.'
ATTR_STEP_CONVERT_MS_PREFIX = 'step.convert_ms.'
# The two parts of a camera frame's age at the step's read: the wait from the driver's hand-off on the runtime clock,
# and the driver's own time from capture to hand-off on the camera clock.
ATTR_STEP_FRAME_WAIT_MS_PREFIX = 'step.frame_wait_ms.'
ATTR_STEP_FRAME_DRIVER_MS_PREFIX = 'step.frame_driver_ms.'

# One episode's waypoint account, totalled over its command channels. ``DROPPED`` counts a waypoint that
# came due and went out on no round.
ATTR_WAYPOINTS_SCHEDULED = 'episode.waypoints.scheduled'
ATTR_WAYPOINTS_EMITTED = 'episode.waypoints.emitted'
ATTR_WAYPOINTS_DROPPED = 'episode.waypoints.dropped'
ATTR_WAYPOINTS_LATE_SUM_MS = 'episode.waypoints.late_sum_ms'
ATTR_WAYPOINTS_LATE_MAX_MS = 'episode.waypoints.late_max_ms'
# The total and the longest time in milliseconds that the arm holds its last target between two chunks.
ATTR_WAYPOINTS_HOLD_SUM_MS = 'episode.waypoints.hold_sum_ms'
ATTR_WAYPOINTS_HOLD_MAX_MS = 'episode.waypoints.hold_max_ms'

# One command that a driver takes: from the tick that reads it to the moment the driver makes it the device's target.
SPAN_DRIVER_COMMAND = 'driver.command'
# On `driver.command`: true for a call whose asker waits for the arrival, false for a streamed setpoint.
ATTR_COMMAND_SYNC = 'command.sync'
# On `driver.command` of a streamed setpoint: the milliseconds from its emit to its read, on the host's monotonic clock.
# A call carries no emit time, so its span has no age.
ATTR_COMMAND_AGE_MS = 'command.age_ms'
