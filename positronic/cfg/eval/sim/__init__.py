import configuronic as cfn

from positronic.cfg.eval.sim import libero, positronic, robolab


@cfn.config(groups=[libero.all, robolab.benchmark, positronic.stack_cubes])
def all(groups):
    """Every sim benchmark in one command: each eval's World in turn, against one warm policy.

    Each entry is itself a list of evals, so a group that grows a second embodiment joins this sweep without
    touching it. Nothing runs concurrently — the evals rebuild the World one at a time, so only the env
    server of the eval in flight is alive.
    """
    return [ev for group in groups for ev in group]
