"""The gateway's paths — one definition the client and the server both bind to.

Every write is a POST, every read a GET, all under `/v1`. Field names live in `requests`, so no
parameter is spelled out here a second time as a string.
"""

from __future__ import annotations

API_PREFIX = '/v1'

USERS_REGISTER = f'{API_PREFIX}/users.register'
USERS_ME = f'{API_PREFIX}/users.me'
# The sign-in the platform takes. It needs no key, so a caller reads it before it holds one.
AUTH_CONFIG = f'{API_PREFIX}/auth.config'
SUBMISSIONS_CREATE = f'{API_PREFIX}/submissions.create'
SUBMISSIONS_RESOLVE = f'{API_PREFIX}/submissions.resolve'
SUBMISSIONS_LIST = f'{API_PREFIX}/submissions.list'
SUBMISSIONS_GET = f'{API_PREFIX}/submissions.get'
SUBMISSIONS_CANCEL = f'{API_PREFIX}/submissions.cancel'
SUBMISSIONS_ARTIFACTS = f'{API_PREFIX}/submissions.artifacts'
RANKINGS_GET = f'{API_PREFIX}/rankings.get'
RANKINGS_LIST = f'{API_PREFIX}/rankings.list'
# The published leaderboards. Every one of these routes needs no key.
LEADERBOARDS_LIST = f'{API_PREFIX}/leaderboards.list'
LEADERBOARDS_GET = f'{API_PREFIX}/leaderboards.get'
LEADERBOARDS_CHARTS = f'{API_PREFIX}/leaderboards.charts'
LEADERBOARDS_EPISODES = f'{API_PREFIX}/leaderboards.episodes'
LEADERBOARDS_PROGRESS = f'{API_PREFIX}/leaderboards.progress'
LEADERBOARDS_MODELS = f'{API_PREFIX}/leaderboards.models'
# What the platform offers the caller: the evals a plan may name, and the tasks it may compose.
CATALOG_EVALS = f'{API_PREFIX}/catalog.evals'
CATALOG_TASKS = f'{API_PREFIX}/catalog.tasks'

BILLING_ACCOUNT = f'{API_PREFIX}/billing.account'
BILLING_PURCHASES_CREATE = f'{API_PREFIX}/billing.purchases.create'
BILLING_PURCHASES_GET = f'{API_PREFIX}/billing.purchases.get'
BILLING_PURCHASES_LIST = f'{API_PREFIX}/billing.purchases.list'
