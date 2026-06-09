import configuronic as cfn
import pos3

from positronic.eval import run_cfg
from positronic.utils.logging import init_logging


# The `positronic` parent command. Only `eval` lives under it today; `inference`/`server`
# fold in as sibling groups in a later, mechanical follow-up.
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({'eval': {'run': run_cfg}})


if __name__ == '__main__':
    _internal_main()
