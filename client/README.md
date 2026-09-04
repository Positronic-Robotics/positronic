# positronic-platform-client

The wire contract for the Positronic evaluation platform, and a thin HTTP client over it: request
and response models, id and enum types, the error envelope, and one `PlatformClient` method per API
endpoint.

> **Alpha, under rapid development.** Names, fields, endpoints and behaviour change without notice,
> and nothing here is covered by a backwards-compatibility guarantee. Pin the exact version you
> tested against, and expect to edit your code when you move off it.

The library depends on `pydantic` and `httpx` and nothing else, so a service that only speaks to the
platform installs it on its own, at the exact version it was written against:

```bash
uv add "positronic-platform-client==0.4.0"
```

`platform_client` never imports `positronic`. Registration ships here (below); the commands that
drive an eval — `positronic eval run` and `positronic account` — ship with `positronic`, which
depends on this package rather than the other way round.

## Registering

This package carries one command, `platform-register`, and it mints the API key everything else
needs. It runs GitHub's device flow: it prints a short code and a URL, waits while you authorize it
in a browser, and registers the account GitHub then names. Installing the package above puts it on
your path.

```bash
export POSITRONIC_PLATFORM_GITHUB_CLIENT_ID=<the OAuth app's public client id>
platform-register --alias=<display name>
export POSITRONIC_PLATFORM_API_KEY=<the key printed above>
```

The token GitHub mints carries `read:user user:email`, so it reads your profile and your verified
email. The platform reads the account once, mints a key, and keeps no GitHub token.

A second run of the command returns the same account and mints no key, because the platform
cannot read back the key it issued. `platform-register --rotate` mints a fresh one, which is how
a machine that no longer holds the key gets one.

`POSITRONIC_PLATFORM_URL` and `--platform-url` name a platform other than the default one, as
everywhere else in this package. The command refuses a plain `http` platform that is not loopback;
the tailnet carries staging with no TLS, so a staging user passes `--plaintext-http`.

## From the command line

The rest of the user-facing side is `positronic`, and from a checkout it needs no installation step
at all. `eval run` runs an eval here when given a policy, and on the platform when given a policy
image. `account register` registers with a credential you already hold, where `platform-register`
mints one from GitHub:

```bash
export POSITRONIC_PLATFORM_CREDENTIAL=<the identity to register with>
uv run positronic account register --alias=<display name>
export POSITRONIC_PLATFORM_API_KEY=<the key printed above>

uv run positronic eval run --eval=<name> --policy-image=org/policy@sha256:…
uv run positronic eval status --submission-id=<hex id>
uv run positronic eval list
uv run positronic eval cancel --submission-id=<hex id>
```

`positronic/cli/examples/` runs the whole flow end to end.

## Configuration

Calls go to `https://platform.positronic.ro` with nothing set. The environment carries the rest:

| Variable | Holds |
|---|---|
| `POSITRONIC_PLATFORM_URL` | a platform other than the default one, overridden per call by `--platform-url` |
| `POSITRONIC_PLATFORM_API_KEY` | the key `register` mints — read from the environment only, so it never reaches a process listing |
| `POSITRONIC_PLATFORM_CREDENTIAL` | the identity `register` registers with — read the same way, for the same reason |

Boards have no command yet — `PlatformClient.list_boards` and `.rankings` read them from Python, the
latter taking a `BoardRef` (`platform_client.boards`). Both take the key when one is set and work
without: a public board is readable by anyone, a tenant's board only by its members.

A board row reads `<display name>#<tag>`. The name is an alias and is not unique — a board may hide
it altogether — so the tag is what tells two rows apart, and it is how you find your own: it is the
same on every board you appear on.

## From Python

```python
from platform_client.client import PlatformClient
from platform_client.evals import EvalRef
from platform_client.policy_images import PolicyImage
from platform_client.requests import SubmissionCreateRequest

with PlatformClient(api_key=key) as client:
    created = client.create_submission(
        SubmissionCreateRequest(policy_image=PolicyImage('org/policy:v1'), eval=EvalRef('robolab.public_subset'))
    )
    view = client.get_submission(created.submission_id)
```

An **eval** is the whole of what a submission chooses: it names a task suite and the embodiment that
runs it — one simulator, or one real robot — so there is no second axis to get wrong. The platform
owns the set of names; asking for one it does not offer raises a `PlatformError` whose `evals`
carries the ones it does.

Ids are 64-bit ints in Python and bare lowercase hex on the wire; closed sets are `IntEnum`s carried
as slugs. A non-2xx response raises `PlatformError`, which carries the parsed error envelope: `code`
for a program, `message` for a human, `reason_code` where a terminal caller fault has one, `quota`
where a 429 names the rule that refused the request, and `evals` where the eval asked for is not one
of them.

`users.me` reports the plan's rules as a list of `QuotaLimit`, each with its own key, window and
subject; `MeResponse.quota_for(QUOTA_SUBMISSIONS_DAY)` reads one by key, from the keys the package
publishes beside `QuotaLimit`.
