import configuronic as cfn
from eval_vocabulary.episode import ABC_TASK_EVAL_SIGNAL

from positronic.cfg.eval import build_tasks, spec
from positronic.drivers.roboarm.models import bundled_yam_model
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
    camera_view=mapping.CameraView.ZED_X,
    # ABC's actions are 34 ms apart. One more period lets the last action's verdict arrive before the deadline.
    timeout=(mapping.ABC_EVAL_STEPS + 1) * 0.034,
    seed=None,
    task=None,
    trial_count=1,
)
def all_tasks(task, trial_count, timeout, camera_dict, camera_height, camera_width, camera_view, seed):
    """An ABC eval on the bimanual i2rt YAM.

    ``task`` is a name, an alias, a prompt, or a list of them; unbound, it sweeps every task ABC lists.
    ``camera_view`` is a ``mapping.CameraView`` or its value.

    A task's instruction is the prompt ABC lists for it. A seeded reset draws the prompt of some tasks, and some
    tasks move the prompt during an episode: the env publishes each such prompt as a new goal.
    """
    view = mapping.CameraView(camera_view)
    proxy = RemoteEnvControlSystem(AbcAdapter(camera_dict), serve_abc(mapping.task_names(task)))
    embodiment = remote_embodiment(
        proxy, camera_dict, descriptor='remote.abc.yam_bimanual', arms=mapping.ARMS, static_meta=bundled_yam_model()
    )
    privileged = {
        mapping.OBS_SIM_STATE: proxy.privileged[mapping.OBS_SIM_STATE],
        ABC_TASK_EVAL_SIGNAL: proxy.privileged[mapping.OBS_TASK_EVAL],
    }

    def tasks() -> list[Task]:
        task_scenes = [
            (
                Task(instruction=listed.instruction, timeout_sec=timeout),
                {
                    **listed.params,
                    abc_keys.CAMERA_HEIGHT: camera_height,
                    abc_keys.CAMERA_WIDTH: camera_width,
                    abc_keys.CAMERA_VIEW: view.value,
                },
            )
            for listed in proxy.tasks(spec(**{mapping.SELECT_TASKS: task}))
        ]
        return build_tasks(task_scenes, seed, trial_count)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


put_bottles = all_tasks.override(task='put_plastic_bottles_in_bin')
