from configuronic.cli import CommandTree

from positronic.cli.account.register import register

# Subcommands of `positronic account`: what the platform knows about you rather than about any one
# run. Running an eval on it, and reading back what a run did, are `positronic eval`.
commands: CommandTree = {'register': register}
