"""Canonical span names and attribute keys of the eval telemetry sidecars.

These literals are the producer↔reducer contract: a producer opens a span (or stamps an attribute) by
them and the offline reduce (``positronic.cli.eval.timing_report``) matches on the same ones. They are
defined here once, in a leaf module with no positronic imports, so a rename is a single-site change the
type checker propagates instead of a string literal duplicated across the harness, the eval CLI, the sim
adapters and the report.
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

ATTR_EPISODE_INDEX = 'episode.index'
ATTR_EPISODE_STEPS = 'episode.steps'
ATTR_EPISODE_VIRTUAL_S = 'episode.virtual_s'
ATTR_EPISODE_ABORTED = 'episode.aborted'
ATTR_EPISODE_PARTIAL = 'episode.partial'
ATTR_PASS_FAILED = 'pass.failed'

# The harness process's sidecar name — the discriminator between client-side spans (episode, client env.step)
# and an env server's own file, which reduces rely on.
HARNESS_PROCESS = 'harness'
