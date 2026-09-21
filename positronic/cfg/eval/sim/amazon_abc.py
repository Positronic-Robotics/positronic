import configuronic as cfn

from positronic import keys
from positronic.cfg.eval import build_tasks, spec
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.models import DEFAULT_FRAME
from positronic.eval import Eval, Observation, Task
from positronic.simulator.amazon_abc import keys as abc_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import AbcAdapter
from positronic.simulator.amazon_abc.launcher import serve_abc
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_embodiment


@cfn.config(
    camera_dict={keys.EXTERIOR_IMAGE: 'top', keys.WRIST_LEFT_IMAGE: 'left', keys.WRIST_RIGHT_IMAGE: 'right'},
    camera_height=168,
    camera_width=224,
    # ABC's own sim-eval budget is 3540 actions of ~34 ms.
    timeout=120.0,
    seed=None,
    task=None,
    trial_count=1,
)
def _abc_eval(task, trial_count, timeout, camera_dict, camera_height, camera_width, seed):
    """An ABC eval on the bimanual i2rt YAM: the embodiment proxies a remote ABC env, the task carries the scene.

    ``task`` selects from ABC's catalogue by canonical name, alias or prompt, and takes a list to sweep several;
    unbound it sweeps every task ABC lists. The assets a task's scene loads are downloaded before the server
    starts, so an unbound ``task`` fetches all of them.

    The instruction is never pinned: the task reads its language live from the env, which reports it in every
    reset's meta — ABC draws a fresh directive per episode on the tasks that carry one. The per-trial seed
    draws the world the trial runs in.

    Each arm reports ``robot_state.{side}`` and ``grip.{side}`` and takes ``robot_command.{side}`` and
    ``target_grip.{side}``, the flat per-arm names the real ``yam_bimanual`` embodiment shares. ABC's full
    physics state is the privileged ground truth (recorded, never fed to the policy), so success is
    recomputable downstream; the live success also rides the trial's terminal.
    """
    selection = None if task is None else [task] if isinstance(task, str) else list(task)
    proxy = RemoteEnvControlSystem(AbcAdapter(camera_dict), serve_abc(selection))
    embodiment = remote_embodiment(
        proxy,
        camera_dict,
        descriptor='remote.abc.yam_bimanual',
        arms=mapping.ARMS,
        # ABC measures and drives each arm at the site the YAM driver names ``DEFAULT_FRAME``.
        static_meta={roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME},
    )
    privileged = {mapping.OBS_SIM_STATE: Observation(proxy.privileged[mapping.OBS_SIM_STATE], None)}
    task_spec = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=timeout)

    def tasks() -> list[Task]:
        scenes = [
            {**params, abc_keys.CAMERA_HEIGHT: camera_height, abc_keys.CAMERA_WIDTH: camera_width}
            for params in proxy.tasks(spec(**{mapping.SELECT_TASKS: task}))
        ]
        return build_tasks(task_spec, seed, trial_count, scenes)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


# The task the integration is built against: two to six plastic bottles into a bin, scored by ABC's own
# evaluator against the bin's measured interior.
put_bottles = _abc_eval.override(task='put_plastic_bottles_in_bin')
