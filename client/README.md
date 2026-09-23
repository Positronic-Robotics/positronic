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
uv add "positronic-platform-client==0.9.0"
uv add "positronic-platform-client @ git+https://github.com/Positronic-Robotics/positronic@<tag or commit>#subdirectory=client"
```

The package is not on PyPI yet. Until it is, use the second line, pinned to a tag or a commit. The
platform serves the same contract as an OpenAPI schema at `<platform>/openapi.json`, and `/docs`
browses it.

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
  - name: baseline                        # remote: the caller's own server
    wire: websocket_tls                   # names the wire, then that wire's address
    address: {host: baseline.example, port: 443, path: /api/v1/session, query: mode=native}
  - name: candidate                       # served: the platform brings it up and records the address
    kind: served
    spec: dreamzero
    wire: grpc
episodes_per_endpoint: 10
episodes_total: 22
cap_per_episode_sec: 180
max_cap_per_episode_sec: 300
policy_preset: example_candidate
tote_placement: random                   # left | right | random | none
external_cameras: {side: random}         # per mount, by the task's name for it
```

An endpoint says where its policy comes from and the wire a session with it runs over. `wire` is a
name from `positronic_wire.registry`: `websocket`, `websocket_tls`, `websocket_unix`, `grpc`,
`grpc_tls` or `roboarena`. A `remote` endpoint then carries the `address` that wire dials, and each
wire declares its own fields: `host`, `port`, `path` and `query` on the websocket and gRPC wires;
`uds`, `path` and `query` on `websocket_unix`; `host` and `port` on `roboarena`. `path` is the
session route (`/api/v1/session`, or `/api/v1/session/<model>`), and `query` defaults to empty. A
`served` endpoint names the `spec` the platform brings up, and the platform records the address. An
`image` endpoint names the container the platform runs. Every kind names its wire, and there is no
default. A record carries no URL: a `url` field is refused, and so is a scheme in `host`. The kinds
refuse each other's fields, so an entry cannot carry two answers to the same question.

`positronic eval run --from-file` files that plan with `submissions.create`. The file is YAML or
JSON, and an `--eval` value is a name. Two or more endpoints make one blind sample: the operator is
told no policy, and each episode records which one served it. `eval status` and `eval list` read it back by
the submission id every run carries. The platform records the plan, the rollouts coordinator runs
it on the lab rig, and a `blocked` run waits on what its `reason` names. A plan that states its own
tasks needs a customer grant; a key without one is refused `forbidden`, and so is `catalog.tasks`.
Write to hi@phail.ai for a grant. A rig plan queues for an operator, so it answers `pending` with a
`queue_position`, and it does not count against the `submissions.day` quota: that quota counts the
image runs the platform executes itself. `users.me` names the grant's client in `client`.

Each run of a rig plan carries `episodes`: what the run took on, and what it recorded. `done` moves
as the rig records each episode. A finished rig plan also carries `replay`, a page that plays back its
episodes, and `outcome`, the kept, judged and successful episodes per endpoint.

The answer to `submissions.create` carries `resolved`, the plan as the rig runs it:
`episodes_total`, and for each task the count per endpoint, the cap, the preset, each side, the
vantage, the clutter objects and the episode order. A level states a value, else the task's
catalogue entry gives it, else the platform draws it. The platform makes each draw once per plan, so every run of
the plan lays out that scene and that table, and runs the episodes in that order.
`submissions.get` carries the same `resolved`. A rig plan whose task resolves no
`cap_per_episode_sec` or no `policy_preset` at any level is refused `bad_request`, and the refusal
names each task and what it lacks.

`submissions.resolve` takes the same plan and answers with `resolved` alone. It files nothing,
spends no quota and returns no submission id. A plan with a `transaction_key` draws from that key,
so a dry run shows the draws a submission under the same key then makes. Without a key, the draws
are an example. From Python, `PlatformClient.resolve_plan` makes the call.

`EvalPlan` refuses unknown fields. `EvalPlan.model_validate(plan)` raises on one before anything
reaches the platform.

A plan states its own tasks and endpoints. A policy image run names a catalog eval:
[Submit a policy image](../docs/submit-a-policy-image.md) says what the platform requires of the
image, and how to build, test and submit it.

A policy image is one endpoint of a plan: `--policy-image` states an `image` endpoint,
`--policy-wire` names the wire the image serves, and `--eval` names the eval whose tasks it runs.
`plan_of_image` builds that shape.

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
when given a plan file. From zero, `platform-register` is the one path to a key: it runs GitHub's
device flow and prints the `export` line. In a checkout, run it through `uv run`. `account register`
takes a GitHub token the platform's OAuth app minted, in `POSITRONIC_PLATFORM_CREDENTIAL`, and saves
the key in the record the other commands read; a new user holds no such token.

```bash
platform-register --alias=<display name>            # in a checkout: uv run platform-register
export POSITRONIC_PLATFORM_API_KEY=<the key it printed>

uv run positronic eval run --eval=<name> --policy-image=org/policy@sha256:… --policy-wire=websocket
uv run positronic eval run --from-file=positronic/cli/examples/rig_plan.yaml
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
| `POSITRONIC_PLATFORM_CREDENTIAL` | a GitHub token the platform's OAuth app minted, for `account register` — read the same way, for the same reason. `platform-register` mints that token itself and needs none |

The record's key and its platform are read one at a time. A command that names another platform, and
no key, sends the record's key to the platform it names. The client speaks one wire contract and runs
against any platform that serves it, but it is built for ours. You take the risk of another one.

`rankings.list` names the boards, and `rankings.get?board=<slug>` reads one. From the command line,
`positronic/cli/examples/nebius_competition/standings.py` prints the boards, or the rows of one:

```bash
uv run positronic/cli/examples/nebius_competition/standings.py
uv run positronic/cli/examples/nebius_competition/standings.py --board=<slug>
```

From Python, `PlatformClient.list_boards` and `.rankings` read them, the latter taking a `BoardRef`
(`platform_client.boards`). Both take the key when one is set and work without: a public board is
readable by anyone, a tenant's board only by its members.

A board row reads `<display name>#<tag>`. The name is an alias and is not unique — a board may hide
it altogether — so the tag is what tells two rows apart, and it is how you find your own: it is the
same on every board you appear on.

## From Python

```python
from platform_client.client import PlatformClient
from platform_client.eval_plan import plan_of_image
from platform_client.policy_images import PolicyImage

with PlatformClient(api_key=key) as client:
    boards = client.list_boards().boards  # public; each names the eval it ranks
    for board in boards:
        print(board.board, board.eval, board.primary_metric)
    created = client.create_submission(plan_of_image(PolicyImage('org/policy:v1'), boards[0].eval))
    view = client.get_submission(created.submission_id)
```

The platform resolves the image reference to a digest at submission and records it as
`policy_image_digest`, so a tag is safe for the run: the run uses the bytes the tag named at that
moment. Pin by digest to know which image was scored.

An **eval** is the whole of what a submission chooses: it names a task suite and the embodiment that
runs it — one simulator, or one real robot — so there is no second axis to get wrong. The platform
owns the set of names. Do not copy a name from a document; read it from the platform:

- `rankings.list` needs no key. Each board it returns names the eval it ranks.
- `submissions.create` with a name the platform does not offer raises a `PlatformError` whose
  `evals` carries every eval on offer, with a board or without one.

Ids are 64-bit ints in Python and bare lowercase hex on the wire; closed sets are `IntEnum`s carried
as slugs. A non-2xx response raises `PlatformError`, which carries the parsed error envelope: `code`
for a program, `message` for a human, `reason_code` where a terminal caller fault has one, `quota`
where a 429 names the rule that refused the request, and `evals` where the eval asked for is not one
of them.

`submissions.get` answers one model per status. A finished run reads its score off `scores` and its
outputs off `artifacts`. A failed run reads `reason_code` and `reason`, and `artifacts` links the
records the run wrote: `result` is its outcome, and `diagnostics` says why its policy failed.
`policy_log` is the policy container's own output, and a finished or failed run carries it. Each
link is a signed URL, because the submitter holds no credential for the bucket. `artifacts` is
absent on a run that wrote nothing, `diagnostics` on a run whose record was not written, and
`policy_log` on a run whose container printed nothing.

`submissions.artifacts` lists what a finished run wrote, one page at a time. Each entry names its
`key` under the submission's own prefix, its `size`, and a signed `url` that expires. Pass `prefix`
to keep the page to one part of the tree, so one attempt's episodes are `attempts/<n>/episodes/`
and `result.json`'s `attempt_location` names the attempt that scored. Pass `after` with the `next`
of the page before it to read the rest. The route answers a finished run alone: a run still going
has a part-written prefix, and a run that failed reads its records off `submissions.get`.

`users.me` reports the plan's rules as a list of `QuotaLimit`, each with its own key, window and
subject; `MeResponse.quota_for(QUOTA_SUBMISSIONS_DAY)` reads one by key, from the keys the package
publishes beside `QuotaLimit`.
