import configuronic as cfn
import pos3

from positronic.cfg.eval import run
from positronic.utils.logging import init_logging


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({'eval': {'run': run}})


if __name__ == '__main__':
    _internal_main()
