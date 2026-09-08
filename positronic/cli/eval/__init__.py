from configuronic.cli import CommandTree

from positronic.cli.eval.run import run
from positronic.cli.eval.submissions import cancel, catalog, list_runs, status
from positronic.cli.eval.timing_report import timing_report

# Subcommands of `positronic eval`: `run` executes an eval, and the rest read back what it did.
commands: CommandTree = {
    'run': run,
    'status': status,
    'list': list_runs,
    'cancel': cancel,
    'catalog': catalog,
    'timing-report': timing_report,
}
