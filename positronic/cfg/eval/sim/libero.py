import configuronic as cfn

from positronic import keys
from positronic.cfg.eval import build_tasks, spec
from positronic.drivers.roboarm.models import bundled_panda_model
from positronic.eval import Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.libero.adapter import LiberoAdapter
from positronic.simulator.libero.launcher import serve_libero


@cfn.config(
    camera_dict={keys.EXTERIOR_IMAGE: 'agentview_image', keys.WRIST_IMAGE: 'eye_in_hand_image'},
    camera_resolution=256,
    control_mode='ee',
    timeout=20.0,
    seed=None,
    task_id=None,
    trial_count=1,
    settle_steps=10,
)
def _libero_eval(
    suite, task_id, trial_count, timeout, camera_dict, camera_resolution, control_mode, seed, settle_steps
):
    """A LIBERO eval: the embodiment proxies a remote LIBERO env, the task carries the scenario.

    A LIBERO *suite* is a set of related manipulation tasks; ``task_id`` selects one within it — a fixed scene and
    language goal (see https://github.com/Lifelong-Robot-Learning/LIBERO). The suites bound below:

      - ``spatial`` (libero_spatial) — same object and goal, varied placement (spatial generalization)
      - ``object`` (libero_object) — different objects, same pick-and-place (object generalization)
      - ``goal`` (libero_goal) — same objects, different goals (goal/procedure generalization)
      - ``libero_10`` (libero_10 / LIBERO-LONG) — long-horizon, entangled tasks across diverse scenes
      - ``libero_90`` (libero_90) — the large short-horizon pool; with libero_10 it forms LIBERO-100

    ``_libero_eval`` leaves ``suite`` unbound; each named config below is a ``.override`` binding it — to one
    suite, or to a list (``all``) swept in one run. An unbound ``task_id`` sweeps every task of each bound suite;
    ``--eval.task_id`` pins one. The env resolves that selection when the run starts, so a suite or a
    ``task_id`` LIBERO does not hold stops the run after the server boots. The instruction is never pinned: the
    task reads its language live from the env, which reports it (with ``suite`` and ``task_id``) in every
    reset's meta.

    positronic launches a single task-agnostic env server in its own 3.10 interpreter; the proxy drives it over
    the socket and the whole scene spec — suite, task_id, camera_resolution, control_mode, settle_steps — rides
    each trial's reset token, so one embodiment serves any mix of suites and tasks. The per-trial seed selects a
    saved init-state and re-randomizes the scene.
    LIBERO's full physics state is the privileged ground truth (recorded, never fed to the policy), so
    success is recomputable downstream; the live ``done`` flag also rides the trial's terminal.
    """
    proxy = RemoteEnvControlSystem(LiberoAdapter(camera_dict), serve_libero())
    # LIBERO drives the same Franka Panda as the native sim, so recordings carry the same model (URDF + meshes +
    # joint names + control frame) for the 3D viewer and offline IK, supplied here since the 3.10 server can't
    # import positronic to emit it via ``robot_meta``.
    embodiment = remote_franka_embodiment(
        proxy, camera_dict, descriptor='remote.libero.franka', static_meta=bundled_panda_model()
    )
    privileged = {'sim_state': Observation(proxy.privileged['sim_state'], None)}
    # rules-allow: hardcoded-keys — the env names this reset-meta field; it is not positronic's ``keys.TASK``
    task = Task(instruction_source=lambda: proxy.meta['task'], timeout_sec=timeout)

    def tasks() -> list[Task]:
        scenes = [
            {
                **params,
                keys.EVAL_CAMERA_RESOLUTION: camera_resolution,
                keys.EVAL_CONTROL_MODE: control_mode,
                keys.EVAL_SETTLE_STEPS: settle_steps,
            }
            for params in proxy.tasks(spec(suite=suite, task_id=task_id))
        ]
        return build_tasks(task, seed, trial_count, scenes)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


# Each suite binds only its LIBERO ``suite``; an unbound ``task_id`` sweeps the whole suite, ``--eval.task_id``
# pins one. The instruction is not pinned — the task reads it live from the env's reset meta.
spatial = _libero_eval.override(suite='libero_spatial')
object = _libero_eval.override(suite='libero_object')
goal = _libero_eval.override(suite='libero_goal')
libero_10 = _libero_eval.override(suite='libero_10')
libero_90 = _libero_eval.override(suite='libero_90')

# The four-suite LIBERO benchmark in one run — the set papers report as "LIBERO average". libero_90 stays out:
# it is the LIBERO-100 pretraining pool, not a standard eval target.
all = _libero_eval.override(suite=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'])
