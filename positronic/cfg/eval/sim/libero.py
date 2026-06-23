import configuronic as cfn

from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.models import bundled_panda_model
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.libero.adapter import LiberoAdapter
from positronic.simulator.libero.launcher import LiberoServer


@cfn.config(
    task_id=0,
    camera_dict={'image.agentview': 'agentview_image', 'image.wrist': 'eye_in_hand_image'},
    camera_resolution=224,
    control_mode='ee',
    timeout=20.0,
    seed=None,
)
def _libero_eval(suite, task_id, instruction, timeout, camera_dict, camera_resolution, control_mode, seed):
    """A LIBERO eval: the embodiment proxies a remote LIBERO env, the task carries the scenario.

    A LIBERO suite (``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``, ...) is a set of
    related manipulation tasks; ``task_id`` selects one task within it — a fixed scene and language goal (see
    https://github.com/Lifelong-Robot-Learning/LIBERO). ``_libero_eval`` leaves ``suite`` and ``instruction``
    unbound, so each concrete task is a named ``.override`` (e.g. ``spatial`` below); ``instruction`` must match
    that task's own language.

    positronic launches the env server in its own 3.10 interpreter for ``(suite, task_id)``; the proxy
    drives it over the socket. The per-trial seed selects a saved init-state and re-randomizes the scene.
    LIBERO's full physics state is the privileged ground truth (recorded, never fed to the policy), so
    success is recomputable downstream; the live ``done`` flag also rides the trial's terminal.
    """
    server = LiberoServer(suite, task_id, camera_resolution, control_mode)
    proxy = RemoteEnvControlSystem(LiberoAdapter(camera_dict), server.host, server.port)
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
        control_systems=(server, proxy),
        simulated=True,
    )
    privileged = {'sim_state': Observation(proxy.privileged['sim_state'], None)}
    task = Task(
        instruction=instruction, timeout=timeout, privileged=privileged, seed=seed, reset=proxy.reset, done=proxy.done
    )
    return Eval(embodiment, task)


# TODO: ``instruction`` must match the LIBERO task's own ``language`` (the server reports it in frame-0
# meta) — verify against the suite when wiring a real policy.
spatial = _libero_eval.override(
    suite='libero_spatial',
    task_id=0,
    instruction='Pick up the black bowl between the plate and the ramekin and place it on the plate.',
)
