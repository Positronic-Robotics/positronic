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
uv add "positronic-platform-client==0.7.0"
```

`platform_client` never imports `positronic`. One command ships here, `platform-register`, which
mints a key from GitHub. The commands that drive an eval, `positronic eval run`, `eval status`,
`eval list`, `eval catalog` and `positronic account`, ship with `positronic`, which depends on this
package.

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

## Eval plans

An eval is a list of tasks. The platform offers named evals, and a customer composes one: an
`EvalPlan` names the catalogue tasks to run, the policies (endpoints) to run them on, and the
episodes each endpoint takes on each task. A plan either states its own `tasks` or names an `eval`
the catalogue expands into them; both arrive at the same set. The plan states the count once. A task may override it for that task, and an endpoint may
override it for that endpoint, so a 10 + 10 + 2 round is one plan. The scene fields sit on the
plan and on a task: `tote_placement`, `camera_vantage`, `external_cameras` and `clutter`. An
endpoint states only its count. `episodes_total` is a checksum a caller may state.
`max_cap_per_episode_sec` is the upper bound on every task's cap.

```yaml
tasks:
  - eight-spoons-into-grey-tote          # a bare id takes the plan's endpoints and counts
  - task_id: marker-in-mug               # a mapping overrides for that task alone
    episodes_per_endpoint: 2
    cap_per_episode_sec: 120
    endpoints: [candidate]                    # a list replaces the plan's list for this task
endpoints:
  - name: baseline
    url: wss://baseline.example/ws
  - name: candidate
    url: wss://candidate.example/ws
episodes_per_endpoint: 10
episodes_total: 22
cap_per_episode_sec: 180
max_cap_per_episode_sec: 300
policy_preset: example_candidate
tote_placement: random                   # left | right | random | none
external_cameras: {side: random}         # per mount, by the task's name for it
```

`positronic eval run` files that plan with `submissions.create`. `--from-file` names the file, and
it is the only option that does: an `--eval` value is a name. The same flags state a plan
without a file — `--policy-url` (repeatable, `NAME=URL`), `--tasks`, `--episodes`, `--cap` and
`--preset`. The scene fields come from a plan file; a run stated in flags takes what each task's
catalogue entry gives it. Two or more endpoints make one blind sample: the operator is told no
policy, and each episode records which one served it. `eval status` and `eval list` read it back by
the submission id every run carries. The platform records the plan, the rollouts coordinator runs
it on the lab rig, and a `blocked` run waits on what its `reason` names. A plan that states its own
tasks needs a customer grant; a key without one is refused `forbidden`.

A policy image is one endpoint of a plan: `--policy-image` states an `image` endpoint and names
the eval whose tasks it runs. `plan_of_image` builds that shape.

`positronic eval catalog` prints what the key may name: `catalog.evals` lists the evals a plan
names, and `catalog.tasks` the tasks a plan may compose. Every registered user sees the
evals a submission can name. A customer grant adds the rig's evals and tasks, filtered to the entries
offered to the grant's client.

From Python, `PlatformClient` takes and answers the models in `platform_client.eval_plan` and
`platform_client.catalog`. The rollouts coordinator's request record is a subclass of `EvalPlan`, so the ask has one
definition.

## From the command line

`positronic` carries the other commands, and a checkout needs no installation step. `eval run`
runs an eval here when given a policy, on the platform when given a policy image, and on the lab rig
when given a policy URL. `account register` registers with a credential you already hold and saves
the key it mints; `platform-register` mints one from GitHub and prints it:

```bash
export POSITRONIC_PLATFORM_CREDENTIAL=<the identity to register with>
uv run positronic account register --alias=<display name>

uv run positronic eval run --eval=<name> --policy-image=org/policy@sha256:…
uv run positronic eval run --policy-url=baseline=wss://baseline.example/ws,candidate=wss://candidate.example/ws --tasks=<task id> --episodes=10 --cap=180
uv run positronic eval status --id=<hex id>
uv run positronic eval list
uv run positronic eval cancel --id=<hex id>
uv run positronic eval catalog
```

`positronic/cli/examples/` runs the whole flow end to end.

## Configuration

Calls go to `https://platform.positronic.ro` with nothing set. The environment carries the rest:

| Variable | Holds |
|---|---|
| `POSITRONIC_PLATFORM_URL` | a platform other than the default one, overridden per call by `--platform-url` |
| `POSITRONIC_PLATFORM_API_KEY` | the key `register` mints — read from the environment or the saved record, never an argument, so it reaches no process listing |
| `POSITRONIC_PLATFORM_CONFIG_DIR` | where `positronic account register` saves that record, else `~/.config/positronic-platform` |
| `POSITRONIC_PLATFORM_CREDENTIAL` | the identity `register` registers with — read the same way, for the same reason |

The record's key and its platform are read one at a time. A command that names another platform, and
no key, sends the record's key to the platform it names. The client speaks one wire contract and runs
against any platform that serves it, but it is built for ours. You take the risk of another one.

Boards have no command yet — `PlatformClient.list_boards` and `.rankings` read them from Python, the
latter taking a `BoardRef` (`platform_client.boards`). Both take the key when one is set and work
without: a public board is readable by anyone, a tenant's board only by its members.

A board row reads `<display name>#<tag>`. The name is an alias and is not unique — a board may hide
it altogether — so the tag is what tells two rows apart, and it is how you find your own: it is the
same on every board you appear on.

## From Python

```python
from platform_client.client import PlatformClient
from platform_client.eval_plan import plan_of_image
from platform_client.evals import EvalRef
from platform_client.policy_images import PolicyImage

with PlatformClient(api_key=key) as client:
    created = client.create_submission(
        plan_of_image(PolicyImage('org/policy:v1'), EvalRef('robolab.public_subset'))
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
