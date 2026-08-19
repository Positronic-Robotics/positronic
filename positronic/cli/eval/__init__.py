from configuronic.cli import CommandTree

from positronic.cli.eval.run import run
from positronic.cli.eval.submissions import cancel, list_submissions, status
from positronic.cli.eval.timing_report import timing_report

# Subcommands of `positronic eval`. `run` executes an eval — here, or on the platform when it is
# given a policy image — and the rest read back what a platform run is doing.
commands: CommandTree = {
    'run': run,
    'status': status,
    'list': list_submissions,
    'cancel': cancel,
    'timing-report': timing_report,
}
