from configuronic.cli import CommandTree

from positronic.cli.account.credits import account, buy, get_purchase, list_purchases
from positronic.cli.account.register import register
from positronic.cli.account.terms import accept_terms, terms

commands: CommandTree = {
    'register': register,
    'terms': terms,
    'accept-terms': accept_terms,
    'credits': {'account': account, 'buy': buy, 'get-purchase': get_purchase, 'list-purchases': list_purchases},
}
