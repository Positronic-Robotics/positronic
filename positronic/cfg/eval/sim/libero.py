import configuronic as cfn

from positronic.cfg.eval import build_trials
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.models import bundled_panda_model
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.libero.adapter import LiberoAdapter
from positronic.simulator.libero.launcher import serve_libero

# Task count per LIBERO suite — the config expands an unbound ``task_id`` over ``range(num_tasks)``. positronic
# cannot import LIBERO (it lives in the 3.10 env server), so the counts are pinned here against the pinned commit.
_SUITE_NUM_TASKS = {'libero_spatial': 10, 'libero_object': 10, 'libero_goal': 10, 'libero_10': 10, 'libero_90': 90}


@cfn.config(
    camera_dict={'image.agentview': 'agentview_image', 'image.wrist': 'eye_in_hand_image'},
    camera_resolution=256,
    control_mode='ee',
    timeout=20.0,
    seed=None,
    task_id=None,
    trial_count=1,
)
def _libero_eval(suite, task_id, trial_count, timeout, camera_dict, camera_resolution, control_mode, seed):
    """A LIBERO eval: the embodiment proxies a remote LIBERO env, the task carries the scenario.

    A LIBERO *suite* is a set of related manipulation tasks; ``task_id`` selects one within it — a fixed scene and
    language goal (see https://github.com/Lifelong-Robot-Learning/LIBERO). The suites bound below:

      - ``spatial`` (libero_spatial, 10 tasks) — same object and goal, varied placement (spatial generalization)
      - ``object`` (libero_object, 10 tasks) — different objects, same pick-and-place (object generalization)
      - ``goal`` (libero_goal, 10 tasks) — same objects, different goals (goal/procedure generalization)
      - ``libero_10`` (libero_10 / LIBERO-LONG, 10 tasks) — long-horizon, entangled tasks across diverse scenes
      - ``libero_90`` (libero_90, 90 tasks) — the large short-horizon pool; with libero_10 it forms LIBERO-100

    ``_libero_eval`` leaves ``suite`` unbound and ``task_id`` ``None``; each suite below is a named ``.override``
    binding only ``suite``. An unbound ``task_id`` sweeps every task in the suite; ``--eval.task_id`` pins one.
    The instruction is never pinned: the task reads its language live from the env, which reports it (with
    ``suite`` and ``task_id``) in every reset's meta.

    positronic launches a single task-agnostic env server in its own 3.10 interpreter; the proxy drives it over
    the socket and ``task_id`` rides each reset token. The per-trial seed selects a saved init-state and re-randomizes
    the scene.
    LIBERO's full physics state is the privileged ground truth (recorded, never fed to the policy), so
    success is recomputable downstream; the live ``done`` flag also rides the trial's terminal.
    """
    proxy = RemoteEnvControlSystem(
        LiberoAdapter(camera_dict, suite=suite, camera_resolution=camera_resolution, control_mode=control_mode),
        serve_libero(),
    )
    observations = {
        'robot_state': Observation(proxy.observations['robot_state'], Serializers.robot_state),
        'grip': Observation(proxy.observations['grip'], None),
        **{logical: Observation(proxy.observations[logical], Serializers.camera_images) for logical in camera_dict},
    }
    commands = {
        'robot_command': Command(proxy.commands['robot_command'], roboarm_command.Reset(), Serializers.robot_command),
        'target_grip': Command(proxy.commands['target_grip'], 0.0, None),
    }
    embodiment = Embodiment(
        descriptor='remote.libero.franka',
        observations=observations,
        commands=commands,
        # LIBERO drives the same Franka Panda as the native sim, so recordings carry the same model (URDF +
        # meshes + joint names + control frame) for the 3D viewer and offline IK, supplied here since the 3.10
        # server can't import positronic to emit it via ``robot_meta``.
        static_meta={**ROBOT_STATIC_META, **bundled_panda_model()},
        meta_source=proxy.robot_meta,
        control_systems=(proxy,),
        simulated=True,
    )
    privileged = {'sim_state': Observation(proxy.privileged['sim_state'], None)}
    task = Task(
        instruction=lambda: proxy.meta['task'],
        timeout=timeout,
        privileged=privileged,
        reset=proxy.reset,
        done=proxy.done,
    )
    # An unbound ``task_id`` sweeps the whole suite; a pinned one runs just that task. Either way it rides each
    # reset token, so the single task-agnostic env server serves every trial.
    task_ids = [task_id] if task_id is not None else list(range(_SUITE_NUM_TASKS[suite]))
    return Eval(embodiment, task, build_trials(seed, trial_count, task_ids))


# Each suite binds only its LIBERO ``suite``; an unbound ``task_id`` sweeps the whole suite, ``--eval.task_id``
# pins one. The instruction is not pinned — the task reads it live from the env's reset meta.

# libero_spatial — 10 tasks
spatial = _libero_eval.override(suite='libero_spatial')

# libero_object — 10 tasks
object = _libero_eval.override(suite='libero_object')

# libero_goal — 10 tasks
goal = _libero_eval.override(suite='libero_goal')

# libero_10 (LIBERO-LONG) — 10 tasks
libero_10 = _libero_eval.override(suite='libero_10')

# libero_90 — 90 tasks
libero_90 = _libero_eval.override(suite='libero_90')
