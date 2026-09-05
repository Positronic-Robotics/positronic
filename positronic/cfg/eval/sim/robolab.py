import configuronic as cfn

from positronic import keys
from positronic.cfg.eval import number_trials, spec
from positronic.eval import Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.adapter import RobolabAdapter
from positronic.simulator.robolab.launcher import serve_robolab


@cfn.config(
    camera_dict={
        keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera',
        keys.EXTERIOR_RIGHT_IMAGE: 'over_shoulder_right_camera',
        keys.WRIST_IMAGE: 'wrist_cam',
    },
    instruction_type='default',
    trial_count=1,
    timeout=None,
)
def _robolab_eval(task, instruction_type, trial_count, timeout, camera_dict):
    """A RoboLab eval: the embodiment proxies a remote RoboLab env, the task carries the scenario.

    RoboLab (https://github.com/NVLabs/RoboLab) is NVIDIA's Isaac Lab benchmark: tabletop manipulation
    tasks on the DROID rig (Franka arm + Robotiq 2F-85), each with a fixed scene, a language instruction in
    three phrasings (``instruction_type`` ``default``/``vague``/``specific``), a time budget of its own and a
    scripted success check. RoboLab scores three task categories — ``visual``, ``relational`` and
    ``procedural`` — and one task can sit on several axes, so the three category sweeps overlap.

    ``_robolab_eval`` leaves ``task`` unbound; each named config below is a ``.override`` binding it — to a
    single task name, a category, or ``None`` for every task. A list of names also works. The env resolves it
    when the run starts. The instruction is never pinned: the task reads its language live from the env, which
    reports the resolved instruction in every reset's meta.

    positronic launches a single task-agnostic env server in RoboLab's own Isaac Lab interpreter; the proxy
    drives it over the socket and the task name + instruction type ride each trial's reset token. There is no
    per-trial seed: RoboLab's eval path exposes no seed hook, so trial params carry none. The env's live
    subtask progress ``[status, completed, total, score]`` is the privileged ground truth (recorded, never
    fed to the policy).
    """
    proxy = RemoteEnvControlSystem(RobolabAdapter(camera_dict), serve_robolab())
    # The DROID rig's model (Franka arm + Robotiq 2F-85) rides the env's ``robot_meta`` — the launcher
    # serializes it for the Isaac Lab server, which cannot build it — so nothing model-specific lives here.
    embodiment = remote_franka_embodiment(proxy, camera_dict, descriptor='remote.robolab.droid')

    def tasks() -> list[Task]:
        trials = []
        for params in proxy.tasks(spec(task=task)):
            # RoboLab truncates an episode at its own budget, so the deadline sits above it and the env's verdict
            # ends a trial.
            deadline = timeout if timeout is not None else params[robolab_keys.EPISODE_LENGTH] + 10.0
            # rules-allow: hardcoded-keys — the env names this reset-meta field; it is not positronic's ``keys.TASK``
            trial = Task(instruction_source=lambda: proxy.meta['task'], timeout_sec=deadline)
            trials += [(trial, {**params, robolab_keys.INSTRUCTION_TYPE: instruction_type}) for _ in range(trial_count)]
        return number_trials(trials)

    return Eval(
        embodiment, tasks, privileged={'subtask': Observation(proxy.privileged['subtask'], None)}, done=proxy.done
    )


# Every task the benchmark holds, in one run.
benchmark = _robolab_eval.override(task=None)

# One category each — the three axes RoboLab reports scores on.
visual = _robolab_eval.override(task='visual')
relational = _robolab_eval.override(task='relational')
procedural = _robolab_eval.override(task='procedural')

# Single-task smoke targets: the simplest pick-and-place, and the task the committed e2e fixture replays.
banana_in_bowl = _robolab_eval.override(task='BananaInBowlTask')
rubiks_cube_and_banana = _robolab_eval.override(task='RubiksCubeAndBananaTask')
