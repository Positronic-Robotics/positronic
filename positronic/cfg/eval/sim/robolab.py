from contextlib import nullcontext

import configuronic as cfn
from eval_vocabulary.episode import GT_STATE_SIGNAL, SUBTASK_SIGNAL

from positronic import keys
from positronic.cfg.eval import env_server_address, number_trials, require_finite_positive, spec
from positronic.drivers.roboarm.models import bundled_yam_model
from positronic.eval import Eval, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_embodiment
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.adapter import RobolabDroidAdapter, RobolabYamAdapter
from positronic.simulator.robolab.launcher import serve_robolab

# The observation name of each YAM camera, as the YAM rig names its cameras.
_YAM_CAMERAS = {
    robolab_keys.TOP_CAMERA: keys.EXTERIOR_IMAGE,
    robolab_keys.LEFT_WRIST_CAMERA: keys.WRIST_LEFT_IMAGE,
    robolab_keys.RIGHT_WRIST_CAMERA: keys.WRIST_RIGHT_IMAGE,
}


def _camera_dict(robot: str, cameras: str) -> dict[str, str]:
    """The wire name each camera of the set takes, keyed as ``camera_dict``.

    A DROID set with one exterior binds it to ``EXTERIOR_IMAGE``, so a two-camera policy reads the same key
    whichever exterior the run renders.
    """
    names = robolab_keys.camera_set(robot, cameras)
    if robot == robolab_keys.YAM:
        return {_YAM_CAMERAS[name]: name for name in names}
    exteriors = [keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2]
    wire = {}
    for name in names:
        wire[keys.WRIST_IMAGE if name == robolab_keys.WRIST_CAMERA else exteriors.pop(0)] = name
    return wire


def _embodiment(robot: str, proxy: RemoteEnvControlSystem, camera_dict: dict[str, str]):
    if robot == robolab_keys.YAM:
        # The env server holds no positronic, so the client carries the arm model, as the rig records it.
        return remote_embodiment(
            proxy,
            camera_dict,
            descriptor='remote.robolab.yam_bimanual',
            arms=keys.BIMANUAL_ARMS,
            static_meta=bundled_yam_model(),
        )
    # The env's ``robot_meta`` carries the DROID rig's model (Franka arm + Robotiq 2F-85).
    return remote_embodiment(proxy, camera_dict, descriptor='remote.robolab.droid')


@cfn.config(
    robot=robolab_keys.DROID,
    cameras=None,
    instruction_type='default',
    trial_count=1,
    timeout=None,
    horizon_sec=None,
    env_server=None,
    batch_size=None,
)
def _robolab_eval(task, instruction_type, trial_count, timeout, horizon_sec, robot, cameras, env_server, batch_size):
    """A RoboLab eval: the embodiment proxies a remote RoboLab env, the task carries the scenario.

    RoboLab (https://github.com/NVLabs/RoboLab) is NVIDIA's Isaac Lab benchmark: tabletop manipulation
    tasks on the DROID rig (Franka arm + Robotiq 2F-85), each with a fixed scene, a language instruction in
    three phrasings (``instruction_type`` ``default``/``vague``/``specific``), a time budget of its own and a
    scripted success check. RoboLab scores three task categories — ``visual``, ``relational`` and
    ``procedural`` — and one task can sit on several axes, so the three category sweeps overlap.

    ``_robolab_eval`` leaves ``task`` unbound; each named config below is a ``.override`` binding it — to a
    single task name, a category, or ``None`` for every task. A list of names also works. The env resolves it
    when the run starts, and lists each task's instruction in the ``instruction_type`` phrasing.

    ``robot`` names the robot in the task scenes, one of ``keys.ROBOTS``: the DROID rig, or the two-arm YAM. The YAM
    takes joint commands only, and its env reports every pose in the world frame of the YAM rig.

    ``cameras`` names the set the run renders, one of ``keys.ROBOT_CAMERA_SETS[robot]``. For the DROID rig: both
    exteriors and the wrist, or one exterior and the wrist. For the YAM: the top camera and both wrists. Without it,
    the run renders ``keys.DEFAULT_CAMERA_SETS[robot]``. RoboLab bakes the set into the registered task, so a run
    holds one set.

    ``horizon_sec`` replaces the episode length of every task of the run, and the default deadline follows it. A
    run with it is not comparable with RoboLab's own scores.

    ``env_server`` is the ``host:port`` of a RoboLab env server that already runs, which this eval shares with
    other evals: each drives one clone of its scene. The server's robot and camera set must be ``robot`` and
    ``cameras``, and every eval
    that shares it must set the same ``horizon_sec``. Without it, the eval launches a server of its own.
    ``batch_size`` is how many evals start each batch on that server; the server builds that many clones for them.
    Without it, a batch waits for every clone of the server.

    positronic launches a single task-agnostic env server in RoboLab's own Isaac Lab interpreter; the proxy
    drives it over the socket and the task name, instruction type and episode length override ride each trial's
    reset token. There is no per-trial seed: RoboLab's eval path exposes no seed hook, so trial params carry none.
    The eval records two privileged signals and never gives them to the policy. ``subtask`` is the env's live
    subtask progress ``[status, completed, total, score]``. ``gt_state`` is the scene state: object poses, gripper
    contacts and progress per condition chain.
    """
    require_finite_positive('timeout', timeout)
    require_finite_positive('horizon_sec', horizon_sec)
    cameras = cameras or robolab_keys.DEFAULT_CAMERA_SETS[robot]
    camera_dict = _camera_dict(robot, cameras)
    if env_server is None:
        serve = serve_robolab(cameras, robot=robot)
    else:
        serve = nullcontext(env_server_address(env_server))
    adapter = RobolabYamAdapter(camera_dict) if robot == robolab_keys.YAM else RobolabDroidAdapter(camera_dict)
    proxy = RemoteEnvControlSystem(adapter, serve, batch_size)
    embodiment = _embodiment(robot, proxy, camera_dict)

    def tasks() -> list[Task]:
        trials = []
        for listed in proxy.tasks(spec(task=task, **{robolab_keys.SELECT_INSTRUCTION_TYPE: instruction_type})):
            params = listed.params
            if horizon_sec is not None:
                params = {**params, robolab_keys.EPISODE_LENGTH_OVERRIDE: horizon_sec}
            # RoboLab truncates an episode at its length, so the deadline sits above it and the env's verdict ends a
            # trial.
            length = params[robolab_keys.EPISODE_LENGTH] if horizon_sec is None else horizon_sec
            deadline = timeout if timeout is not None else length + 10.0
            trial = Task(instruction=listed.instruction, timeout_sec=deadline)
            trials += [(trial, params) for _ in range(trial_count)]
        return number_trials(trials)

    privileged = {
        SUBTASK_SIGNAL: proxy.privileged[robolab_keys.SUBTASK],
        GT_STATE_SIGNAL: proxy.privileged[robolab_keys.GT_STATE],
    }
    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


# Every task the benchmark holds, in one run.
benchmark = _robolab_eval.override(task=None)

# One category each — the three axes RoboLab reports scores on.
visual = _robolab_eval.override(task='visual')
relational = _robolab_eval.override(task='relational')
procedural = _robolab_eval.override(task='procedural')

# Single-task smoke targets: the simplest pick-and-place, and the task the committed e2e fixture replays.
banana_in_bowl = _robolab_eval.override(task='BananaInBowlTask')
rubiks_cube_and_banana = _robolab_eval.override(task='RubiksCubeAndBananaTask')

# The two-arm YAM in the RoboLab task scenes: every task, and the simplest pick-and-place.
yam_benchmark = _robolab_eval.override(task=None, robot=robolab_keys.YAM)
yam_banana_in_bowl = yam_benchmark.override(task='BananaInBowlTask')
