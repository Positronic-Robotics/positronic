import sys

import configuronic as cfn
import pos3

from pimm.logging import init_logging
from positronic.cli.account import commands as account_commands
from positronic.cli.account.gateway import CommandFailed
from positronic.cli.eval import commands as eval_commands


@pos3.with_mirror()
def _internal_main():
    init_logging()
    try:
        cfn.cli({'eval': eval_commands, 'account': account_commands})
    except CommandFailed as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(exc.fault) from None


if __name__ == '__main__':
    _internal_main()
