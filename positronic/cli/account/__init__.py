from configuronic.cli import CommandTree

from positronic.cli.account.credits import account, buy, get_purchase, list_purchases
from positronic.cli.account.register import register

commands: CommandTree = {
    'register': register,
    'credits': {'account': account, 'buy': buy, 'get-purchase': get_purchase, 'list-purchases': list_purchases},
}
