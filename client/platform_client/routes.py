"""The gateway's paths — one definition the client and the server both bind to.

Every write is a POST, every read a GET, all under `/v1`. Only the paths live here: a POST body and
a GET's query parameters are both models in `requests`, so pydantic owns every field name on both
sides of the wire and no parameter is spelled out a second time as a string.
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
