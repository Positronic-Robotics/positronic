import configuronic as cfn

from positronic.cfg.eval.sim import libero, positronic, robolab


@cfn.config(groups=[libero.all, robolab.benchmark, positronic.stack_cubes])
def all(groups):
    """Every sim benchmark in one command, against one warm policy.

    Each entry is itself a list of evals, so a group that grows a second embodiment joins this sweep without
    touching it. Adding a GPU-hungry backend here is safe only while evals run one at a time.
    """
    return [ev for group in groups for ev in group]
