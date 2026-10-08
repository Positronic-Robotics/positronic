from configuronic.cli import CommandTree

from positronic.cli.account.credits import account, buy, get_purchase, list_purchases
from positronic.cli.account.register import register

# Subcommands of `positronic account`: what the platform knows about you rather than about any one
# run. Running an eval on it, and reading back what a run did, are `positronic eval`.
commands: CommandTree = {
    'register': register,
    'credits': {'account': account, 'buy': buy, 'get-purchase': get_purchase, 'list-purchases': list_purchases},
}
