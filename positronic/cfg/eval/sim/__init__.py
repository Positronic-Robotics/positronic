import configuronic as cfn

from positronic.cfg.eval.sim import libero, positronic, robolab


@cfn.config(groups=[libero.all, robolab.benchmark, positronic.stack_cubes])
def all(groups):
    """Every sim benchmark in one command, against one warm policy.

    Each entry is itself a list of evals, so a group that grows a second embodiment joins this sweep without
    touching it. Listing backends together is only safe because ``eval run`` rebuilds the World one eval at a
    time: two env servers never hold the GPU at once.
    """
    return [ev for group in groups for ev in group]
