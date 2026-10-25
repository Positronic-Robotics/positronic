import configuronic as cfn
from eval_vocabulary.episode import ABC_TASK_EVAL_SIGNAL

from positronic.cfg.eval import build_tasks, spec
from positronic.drivers.roboarm.models import GRIPPER_SITE, bundled_yam_model
from positronic.eval import Eval, Task
from positronic.simulator.amazon_abc import keys as abc_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import CAMERAS, AbcAdapter
from positronic.simulator.amazon_abc.launcher import serve_abc
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_embodiment


@cfn.config(
    camera_dict=CAMERAS,
    camera_height=260,
    camera_width=416,
    # ABC's actions are 34 ms apart. One more period lets the last action's verdict arrive before the deadline.
    timeout=(mapping.ABC_EVAL_STEPS + 1) * 0.034,
    seed=None,
    task=None,
    trial_count=1,
)
def all_tasks(task, trial_count, timeout, camera_dict, camera_height, camera_width, seed):
    """An ABC eval on the bimanual i2rt YAM.

    ``task`` is a name, an alias, a prompt, or a list of them; unbound, it sweeps every task ABC lists.
    """
    proxy = RemoteEnvControlSystem(AbcAdapter(camera_dict), serve_abc(mapping.task_names(task)))
    embodiment = remote_embodiment(
        proxy,
        camera_dict,
        descriptor='remote.abc.yam_bimanual',
        pose_site=GRIPPER_SITE,
        arms=mapping.ARMS,
        static_meta=bundled_yam_model(),
    )
    privileged = {
        mapping.OBS_SIM_STATE: proxy.privileged[mapping.OBS_SIM_STATE],
        ABC_TASK_EVAL_SIGNAL: proxy.privileged[mapping.OBS_TASK_EVAL],
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
