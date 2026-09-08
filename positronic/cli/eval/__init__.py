from configuronic.cli import CommandTree

from positronic.cli.eval.run import run
from positronic.cli.eval.submissions import cancel, catalog, list_runs, status
from positronic.cli.eval.timing_report import timing_report

# Subcommands of `positronic eval`. `run` executes an eval — here, on the platform when it is given
# a policy image, or on the rig when it is given a policy URL — and the rest read back what a run
# sent to the platform is doing.
commands: CommandTree = {
    'run': run,
    'status': status,
    'list': list_runs,
    'cancel': cancel,
    'catalog': catalog,
    'timing-report': timing_report,
}
