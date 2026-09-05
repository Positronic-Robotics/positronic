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
uv add "positronic-platform-client==0.5.0"
```

`platform_client` never imports `positronic`. Two commands ship here: `platform-register`, and
`positronic-platform`, which registers and then files and reads rollout requests. The commands that
drive an eval, `positronic eval run` and `positronic account`, ship with `positronic`, which depends
on this package.

## Registering

`platform-register` mints the API key that every other call needs. It runs GitHub's device flow:
it prints a short code and a URL, waits while you authorize the app in a browser, and registers
the GitHub account that authorized it. The package install above puts the command on your path.

```bash
platform-register --alias=<display name>
export POSITRONIC_PLATFORM_API_KEY=<the key the command printed>
```

The command registers through the platform's own OAuth app, whose client id it carries as its
default. `--client-id` and `POSITRONIC_PLATFORM_GITHUB_CLIENT_ID` name another app instead.

The GitHub token carries the scopes `read:user` and `user:email`. The platform reads the account
once, mints a key, and stores no GitHub token.

A second run returns the same account and no key: the platform cannot read back a key it issued.
Run `platform-register --rotate` to mint a new key on a machine that lost it.

`--platform-url` and `POSITRONIC_PLATFORM_URL` name a platform other than the default. The command
refuses a plain `http` platform that is not loopback. Staging has no TLS and is reached over the
tailnet: pass `--plaintext-http` to reach it.

## Rollout requests

A customer files a rollout request: the catalogue tasks to run, the policies to run them on, and
the episodes each policy takes on each task. The count is stated once on the request; a task states
its own for itself, and an endpoint states its own for that endpoint, so a 10 + 10 + 2 round is one
request. The platform records it, the rollouts coordinator files it and runs it, and the request's
status reads back through the same key. A key needs a customer grant for these calls; a key without
one is refused `forbidden`.

```bash
positronic-platform register --alias='<display name>'
positronic-platform requests create --tasks eight-spoons-into-grey-tote \
    --endpoints gyros=wss://host/ws --episodes-per-endpoint 10 --cap 180 --preset runway_ziyi \
    --scene tote_placement=random --scene camera.side=left
positronic-platform requests create --from request.json
positronic-platform requests get <hex id>
positronic-platform requests list --after <hex id> --limit 50
```

`--from` takes a whole `RequestCreate` as JSON, which is how a served endpoint, a per-task override
or a per-endpoint count is filed; the flags cover the common round. `--scene` takes `tote_placement=<side>`,
`camera_vantage=<vantage>` and `camera.<mount>=<side>`, where a side is `left`, `right`, `random`
or `none`. Every command prints its answer as JSON, so an agent reads it back as the models in
`platform_client.responses`.

`register` runs the same GitHub device flow as `platform-register` and then writes
`~/.config/positronic-platform/config.json`, mode 0600, holding the platform's URL and the key
together. The key never appears on a command line: a command reads it from
`POSITRONIC_PLATFORM_API_KEY`, else from the file `--api-key-file` names, else from that record.
The platform is `--platform-url`, else `POSITRONIC_PLATFORM_URL`, else that record, else the default
below. `POSITRONIC_PLATFORM_CONFIG_DIR` names another config directory.

From Python, `PlatformClient.requests_create`, `.requests_get` and `.requests_list` take and answer
the same models; `requests_list` pages oldest first, and a page's `next` is the `after` of the page
after it.

## From the command line

`positronic` carries the other commands, and a checkout needs no installation step. `eval run`
runs an eval here when given a policy, and on the platform when given a policy image.
`account register` registers with a credential you already hold; `platform-register` mints one
from GitHub:

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
