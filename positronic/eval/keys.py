"""The keys a trial writes: what it readies, the conditions it runs under and the verdict it ends on."""

# The names of what a trial readies before it opens. ``Embodiment.prepare_handlers`` is keyed by them, and so
# is what a ``Task`` asks for. A rig with two arms names its arms ``arm.{side}``. ``SCENE`` means the world
# this trial runs in is ready, drawn by whichever handler the embodiment binds.
ARM = 'arm'
GRIPPER = 'gripper'
SCENE = 'scene'

# The 3D viewer's pointers into an episode: which signals carry joint angles, which carry poses. One arm per
# ``JOINT_SIGNALS`` entry, each running ``URDF`` and standing where ``MOUNTS`` places it, keyed by that signal.
JOINT_SIGNALS = 'joint_signals'
POSE_SIGNALS = 'pose_signals'
MOUNTS = 'mounts'

# What a trial reports when it ends, in its episode's statics. The harness writes ``TERMINATED``:
# True when a terminal was delivered inside the budget, False when the budget ran out. ``SUCCESS``
# rides in the terminal payload an env's adapter returns, so an env that reports it only on success
# leaves it absent on failure — a reader defaults it rather than assuming a False.
SUCCESS = 'eval.success'
TERMINATED = 'eval.terminated'
# Whether the trial charged each model call the wall time it really took. True on a real rig whatever the
# task asked, since it cannot hold the world still while the model thinks.
CHARGE_INFERENCE_TIME = 'eval.charge_inference_time'
# Who ended the trial, when it was not the task's own ground truth. An env's terminal leaves it absent.
ENDED_BY = 'eval.ended_by'
ENDED_BY_OPERATOR = 'operator'

# The conditions the trial ran under, stamped into its episode's statics. ``UNIVERSE`` is ``'sim'`` or
# ``'real'``; ``TIMEOUT`` is absent from an episode whose task set no budget.
UNIVERSE = 'eval.universe'
EMBODIMENT = 'eval.embodiment'
TIMEOUT = 'eval.timeout'

# A trial's place in its eval's sweep, stamped into every trial's params and recorded in its episode's
# statics. ``SEED`` also rides an env's reset token, where absent means the env draws its own.
SEED = 'eval.seed'
TRIAL_INDEX = 'eval.trial_index'
TRIAL_COUNT = 'eval.trial_count'

# The id of the task a trial runs, as the benchmark names it. The episode records it; positronic never reads
# inside it.
TASK = 'eval.task'
