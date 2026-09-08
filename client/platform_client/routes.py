"""The gateway's paths — one definition the client and the server both bind to.

Every write is a POST, every read a GET, all under `/v1`. Field names live in `requests`, so no
parameter is spelled out here a second time as a string.
"""

from __future__ import annotations

API_PREFIX = '/v1'

USERS_REGISTER = f'{API_PREFIX}/users.register'
USERS_ME = f'{API_PREFIX}/users.me'
SUBMISSIONS_CREATE = f'{API_PREFIX}/submissions.create'
SUBMISSIONS_LIST = f'{API_PREFIX}/submissions.list'
SUBMISSIONS_GET = f'{API_PREFIX}/submissions.get'
SUBMISSIONS_CANCEL = f'{API_PREFIX}/submissions.cancel'
RANKINGS_GET = f'{API_PREFIX}/rankings.get'
RANKINGS_LIST = f'{API_PREFIX}/rankings.list'
# A plan a caller composes, filed and read back: the rig's door, beside the competition's `submissions.*`.
EVALS_RUN = f'{API_PREFIX}/evals.run'
EVALS_GET = f'{API_PREFIX}/evals.get'
EVALS_LIST = f'{API_PREFIX}/evals.list'
# What the platform offers the caller: the evals `evals.run` takes by name, and the tasks a plan may name.
CATALOG_EVALS = f'{API_PREFIX}/catalog.evals'
CATALOG_TASKS = f'{API_PREFIX}/catalog.tasks'
