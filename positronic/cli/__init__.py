import configuronic as cfn
import pos3

from positronic.cli.eval import commands as eval_commands
from positronic.utils.logging import init_logging


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({'eval': eval_commands})


if __name__ == '__main__':
    _internal_main()
