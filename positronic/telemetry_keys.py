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
ATTR_PASS_FAILED = 'pass.failed'

# What the server reported spending, stamped on `policy.infer` under this prefix — `served.infer_ms`
# and its siblings, straight from the answer's own `timing` block.
ATTR_SERVED_PREFIX = 'served.'

# Which codec a `policy.encode` span timed, and how many bytes the observation took on the wire.
ATTR_CODEC = 'codec'
ATTR_WIRE_BYTES = 'wire.bytes'

# The harness process's sidecar name — the discriminator between client-side spans (episode, client env.step)
# and an env server's own file, which reduces rely on.
HARNESS_PROCESS = 'harness'

# One harness step: it reads the observations, calls the policy and emits the commands. The policy's spans are
# its children. The step's attributes hold its durations in milliseconds, so one span carries the breakdown.
SPAN_HARNESS_STEP = 'harness.step'
# The time from the step's due time to its start: other loops in the process, and a late wake from sleep.
# A step has no value when an answer started it before its due time, and the first step has no due time.
ATTR_STEP_LATE_MS = 'step.late_ms'
ATTR_STEP_OBSERVE_MS = 'step.observe_ms'
ATTR_STEP_POLICY_MS = 'step.policy_ms'
ATTR_STEP_EMIT_MS = 'step.emit_ms'
# One attribute for each observation signal, with the signal's name after the prefix. A read is the receiver's
# read. A conversion is the serializer and the copy of a new message, so a signal with no new message has none.
ATTR_STEP_READ_MS_PREFIX = 'step.read_ms.'
ATTR_STEP_CONVERT_MS_PREFIX = 'step.convert_ms.'

# One episode's waypoint account, totalled over its command channels. ``DROPPED`` counts a waypoint that
# came due and went out on no round.
ATTR_WAYPOINTS_SCHEDULED = 'episode.waypoints.scheduled'
ATTR_WAYPOINTS_EMITTED = 'episode.waypoints.emitted'
ATTR_WAYPOINTS_DROPPED = 'episode.waypoints.dropped'
ATTR_WAYPOINTS_LATE_SUM_MS = 'episode.waypoints.late_sum_ms'
ATTR_WAYPOINTS_LATE_MAX_MS = 'episode.waypoints.late_max_ms'
