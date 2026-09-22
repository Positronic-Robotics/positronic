import configuronic as cfn

from positronic.cfg.eval import build_tasks, spec
from positronic.drivers.roboarm.models import bundled_yam_model
from positronic.eval import Eval, Observation, Task
from positronic.simulator.amazon_abc import keys as abc_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import CAMERAS, AbcAdapter
from positronic.simulator.amazon_abc.launcher import serve_abc
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_embodiment


@cfn.config(
    camera_dict=CAMERAS,
    camera_height=260,
    camera_width=416,
    # ABC's own sim-eval budget is 3540 actions of ~34 ms.
    timeout=120.0,
    seed=None,
    task=None,
    trial_count=1,
)
def all_tasks(task, trial_count, timeout, camera_dict, camera_height, camera_width, seed):
    """An ABC eval on the bimanual i2rt YAM.

    ``task`` is a name, an alias, a prompt, or a list of them; unbound, it sweeps every task ABC lists.
    """
    selection = None if task is None else [task] if isinstance(task, str) else list(task)
    proxy = RemoteEnvControlSystem(AbcAdapter(camera_dict), serve_abc(selection))
    embodiment = remote_embodiment(
        proxy, camera_dict, descriptor='remote.abc.yam_bimanual', arms=mapping.ARMS, static_meta=bundled_yam_model()
    )
    privileged = {
        name: Observation(proxy.privileged[name], None) for name in (mapping.OBS_SIM_STATE, mapping.OBS_TASK_EVAL)
    }
    task_spec = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=timeout)

    def tasks() -> list[Task]:
        scenes = [
            {**params, abc_keys.CAMERA_HEIGHT: camera_height, abc_keys.CAMERA_WIDTH: camera_width}
            for params in proxy.tasks(spec(**{mapping.SELECT_TASKS: task}))
        ]
        return build_tasks(task_spec, seed, trial_count, scenes)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


put_bottles = all_tasks.override(task='put_plastic_bottles_in_bin')
