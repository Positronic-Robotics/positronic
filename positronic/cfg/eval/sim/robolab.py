import configuronic as cfn

from positronic.drivers.roboarm.models import bundled_franka_model
from positronic.eval import Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.robolab.adapter import RobolabAdapter
from positronic.simulator.robolab.launcher import serve_robolab

# The 120 benchmark tasks: name -> (category, episode_length_s). positronic cannot import robolab (it lives in
# the Isaac Lab env server), so the table is pinned here against the launcher's commit (7d45d749), generated
# from robolab/tasks/_metadata/task_metadata.json. The category is the task's first attribute mapped through
# RoboLab's BENCHMARK_TASK_CATEGORIES (attribute -> visual/relational/procedural); ``None`` marks the three
# tasks the metadata leaves untagged — they run only by name or in ``all``.
_TASKS = {
    'AnimalsInBinTask': ('visual', 90),
    'AppleAndYogurtInBowlTask': ('procedural', 120),
    'BBQSauceInBinTask': ('visual', 90),
    'BagelsOnPlateTask': ('visual', 60),
    'BananaInBowlTask': ('visual', 50),
    'BananaOnPlateTask': (None, 40),
    'BananaThenRubiksCubeTask': ('relational', 60),
    'BananasInBinOneMoreTask': ('visual', 60),
    'BananasInBinThreeTotalTask': ('visual', 60),
    'BananasInCrateTask': ('relational', 60),
    'BananasOutOfBinTask': ('visual', 90),
    'BigPumpkinInBinTask': ('visual', 60),
    'BlackItemsInBinTask': ('visual', 120),
    'BlockStackingOrderAgnosticTask': ('procedural', 90),
    'BlockStackingSpecifiedOrderTask': ('procedural', 90),
    'BlocksInBinTask': ('procedural', 150),
    'BowlInBinTask': (None, 60),
    'BowlStackingLeftOnRightTask': ('relational', 20),
    'BowlStackingRightOnLeftTask': ('relational', 20),
    'ButterAboveRaisinTask': ('relational', 40),
    'CannedFoodInBinTask': ('visual', 60),
    'ClampInRightBinTask': ('visual', 60),
    'CleanUpToysTask': ('procedural', 300),
    'ClearOrganicObjectsTask': ('visual', 240),
    'ClutterPlasticTask': ('visual', 180),
    'ClutterPumpkinTask': ('visual', 90),
    'CoffeePotInBinTask': ('visual', 60),
    'CondimentsInBinTask': ('visual', 180),
    'CookingClearPlateTask': ('visual', 180),
    'CookingPickPastaToolTask': ('relational', 60),
    'CubesAndBlocksInBinTask': ('procedural', 240),
    'DishesInBinTask': ('visual', 180),
    'ElectronicsInBinTask': ('visual', 180),
    'FoodPacking1BoxesTask': ('visual', 60),
    'FoodPacking1CansTask': ('visual', 60),
    'FoodPacking2BoxesTask': ('visual', 180),
    'FoodPacking2CansTask': ('visual', 180),
    'FoodPacking3BoxesTask': ('visual', 240),
    'FoodPacking3CansTask': ('visual', 240),
    'FoodPackingByColorTask': ('relational', 120),
    'FruitsGreenLimesOnPlateTask': ('visual', 90),
    'FruitsMovingOrangeOrLimeTask': ('relational', 60),
    'FruitsMovingTask': ('visual', 60),
    'FruitsOnPlate3Task': ('visual', 200),
    'FruitsOnPlateTask': ('visual', 300),
    'FruitsOnionTask': ('visual', 60),
    'FruitsOnionToPlateTask': ('visual', 60),
    'FruitsOrangesOnPlateTask': ('relational', 90),
    'GrabABagelTask': ('visual', 30),
    'GrabAFruitTask': ('visual', 30),
    'GreenSpoonsInPotTask': ('procedural', 180),
    'HammersInLeftBinTask': ('visual', 180),
    'JugsOnShelfTask': ('visual', 120),
    'KeyboardOutOfBinTask': ('relational', 60),
    'LargerObjectRaisinBoxInBinTask': ('visual', 30),
    'MarkerInMugTask': ('procedural', 40),
    'MouseOnKeyboardTask': ('visual', 60),
    'MoveBananaToBagelPlateTask': ('visual', 90),
    'MustardAboveRaisinTask': ('relational', 40),
    'MustardInLeftBinTask': ('relational', 30),
    'MustardInRightBinTask': ('relational', 30),
    'NonHammerToolsInRightBinTask': ('visual', 180),
    'OneBottleInSquarePailTask': ('visual', 60),
    'OneBottleOnShelfTask': ('visual', 60),
    'PhoneOrRemoteInBinTask': ('relational', 60),
    'PickDrillTask': ('visual', 40),
    'PickGlassesTask': ('visual', 30),
    'PickOrangeObjectTask': ('visual', 60),
    'PickUpBluePitcherTask': ('visual', 30),
    'PickUpGreenObjectTask': ('visual', 30),
    'PinkSpoonInPotTask': ('visual', 60),
    'PlasticBottlesInSquarePailTask': ('visual', 180),
    'PutBowlOnShelfTopTask': ('relational', 60),
    'PutMugsOnShelfTask': ('procedural', 180),
    'PutTwoMugsOnShelfTask': ('procedural', 180),
    'RecycleCartonTask': ('visual', 90),
    'RecycleCartonsOnBoxTask': ('visual', 90),
    'RecycleCartonsVerticalCrateTask': ('visual', 90),
    'RedDishesInBinTask': ('visual', 60),
    'RedItemsInBinTask': ('visual', 60),
    'ReorientAllMugsTask': ('procedural', 90),
    'ReorientJugTask': ('visual', 60),
    'ReorientRedMugTask': ('procedural', 60),
    'ReorientWhiteMugsTask': ('procedural', 60),
    'RubiksCubeAndBananaTask': ('relational', 60),
    'RubiksCubeBehindBowlTask': ('relational', 30),
    'RubiksCubeInFrontOfBowlTask': ('relational', 30),
    'RubiksCubeLeftOfBowlTask': ('relational', 30),
    'RubiksCubeOrBananaTask': ('relational', 30),
    'RubiksCubeRightOfBowlTask': ('relational', 30),
    'RubiksCubeTask': (None, 40),
    'RubiksCubeThenBananaTask': ('relational', 60),
    'RubiksCubesInBinTask': ('procedural', 120),
    'SauceBottlesCrateTask': ('visual', 40),
    'SmallPumpkinInBinTask': ('visual', 60),
    'SmallerObjectButterInBinTask': ('visual', 30),
    'SmartphoneInBinTask': ('visual', 60),
    'SpoonInMugTask': ('procedural', 60),
    'SpoonsInPotTask': ('procedural', 180),
    'Stack3RubiksCubeTask': ('procedural', 60),
    'StackWhiteMugsTask': ('procedural', 60),
    'StackYellowOnRedTask': ('procedural', 60),
    'TakeMeasuringSpoonOutTask': ('visual', 40),
    'TakeMugsOffOfShelfTask': ('visual', 180),
    'TakeSpatulaOffShelfTask': ('procedural', 60),
    'ThrowAwayAppleTask': ('visual', 60),
    'ThrowAwaySnacksTask': ('visual', 120),
    'ToolOrganizationBothTask': ('visual', 180),
    'ToolOrganizationTask': ('visual', 180),
    'ToolsPickingAllHammersTask': ('visual', 240),
    'ToolsPickingDrillTask': ('relational', 60),
    'ToolsPickingHammerTask': ('visual', 60),
    'ToyInBinTask': ('visual', 60),
    'UnstackRubiksCubeTask': ('procedural', 90),
    'UtensilsInMugTask': ('visual', 90),
    'WhiteMugInCenterOfTableTask': ('visual', 30),
    'WhiteMugsInBinTask': ('visual', 60),
    'WoodSpatulaToBowlTask': ('visual', 60),
    'YellowAndWhiteObjectsInBinTask': ('visual', 60),
    'YogurtInBowlTask': ('visual', 40),
}

_CATEGORIES = ('visual', 'relational', 'procedural')


def _resolve_tasks(task) -> list[str]:
    """A task name, a list of names, a category, or ``'all'`` -> the benchmark task names to run."""
    if task == 'all':
        return list(_TASKS)
    if task in _CATEGORIES:
        return [name for name, (category, _) in _TASKS.items() if category == task]
    return [task] if isinstance(task, str) else list(task)


@cfn.config(
    camera_dict={'image.exterior': 'over_shoulder_left_camera', 'image.wrist': 'wrist_cam'},
    instruction_type='default',
    trial_count=1,
    timeout=None,
)
def _robolab_eval(task, instruction_type, trial_count, timeout, camera_dict):
    """A RoboLab eval: the embodiment proxies a remote RoboLab env, the task carries the scenario.

    RoboLab (https://github.com/NVLabs/RoboLab) is NVIDIA's Isaac Lab benchmark: 120 tabletop manipulation
    tasks on the DROID rig (Franka arm + Robotiq 2F-85), each with a fixed scene, a language instruction in
    three phrasings (``instruction_type`` ``default``/``vague``/``specific``), and a scripted success check.
    Every task carries a category — ``visual`` (color/size/semantics), ``relational`` (spatial/conjunction/
    counting), or ``procedural`` (stacking/sorting/reorientation/affordance).

    ``_robolab_eval`` leaves ``task`` unbound; each named config below is a ``.override`` binding it — to a
    single task name, a category, or ``all``. A list of names also works. The instruction is never pinned:
    the task reads its language live from the env, which reports the resolved instruction in every reset's
    meta.

    positronic launches a single task-agnostic env server in RoboLab's own Isaac Lab interpreter; the proxy
    drives it over the socket and the task name + instruction type ride each trial's reset token. There is no
    per-trial seed: RoboLab's eval path exposes no seed hook, so trial contexts carry none. The env's live
    subtask progress ``[status, completed, total, score]`` is the privileged ground truth (recorded, never
    fed to the policy).
    """
    names = _resolve_tasks(task)
    if timeout is None:
        # The env truncates itself at each task's episode_length_s; the harness deadline is a backstop above
        # the longest selected task.
        timeout = max(_TASKS[name][1] for name in names) + 10.0
    proxy = RemoteEnvControlSystem(RobolabAdapter(camera_dict), serve_robolab())
    # RoboLab drives the DROID rig — a Franka arm with the Robotiq 2F-85 — so recordings carry the bundled
    # franka model (URDF + meshes + joint names + control frame) for the 3D viewer and offline IK, supplied
    # here since the Isaac Lab server can't import positronic to emit it via ``robot_meta``.
    embodiment = remote_franka_embodiment(
        proxy, camera_dict, descriptor='remote.robolab.droid', static_meta=bundled_franka_model()
    )
    # RoboLab exposes no seed hook, so trial contexts carry no ``eval.seed``.
    trials = [
        {'eval.task': name, 'eval.instruction_type': instruction_type} for name in names for _ in range(trial_count)
    ]
    for i, ctx in enumerate(trials):
        ctx.update({'eval.trial_index': i, 'eval.trial_count': len(trials)})
    return Eval(
        embodiment,
        Task(
            instruction=lambda: proxy.meta['task'],
            timeout=timeout,
            privileged={'subtask': Observation(proxy.privileged['subtask'], None)},
            reset=proxy.reset,
            done=proxy.done,
        ),
        trials,
    )


# The full 120-task benchmark in one run.
benchmark = _robolab_eval.override(task='all')

# One category each — the three axes RoboLab reports scores on.
visual = _robolab_eval.override(task='visual')
relational = _robolab_eval.override(task='relational')
procedural = _robolab_eval.override(task='procedural')

# Single-task smoke targets: the simplest pick-and-place, and the task the committed e2e fixture replays.
banana_in_bowl = _robolab_eval.override(task='BananaInBowlTask')
rubiks_cube_and_banana = _robolab_eval.override(task='RubiksCubeAndBananaTask')
