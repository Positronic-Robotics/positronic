# Examples

Runnable walkthroughs of the evaluation platform. Everything here runs through `uv` from a
checkout, so nothing has to be installed first — `uv run` builds the environment it needs.

| Script | What it shows |
|---|---|
| `walkthrough.py` | The whole flow through `PlatformClient`: register, submit, read quota, poll to a terminal status. |
| `nebius_competition/submit_sample.py` | Submitting to one engagement's eval with a transaction key, and waiting for what it scored. |
| `nebius_competition/standings.py` | The public boards, and the rows of one board. No key. |

## From the command line

The same flow, with no Python of your own:

```bash
platform-register --alias=<display name>            # in a checkout: uv run platform-register
export POSITRONIC_PLATFORM_API_KEY=<the key it printed>

uv run positronic eval run --eval=<name> --policy-image=org/policy@sha256:…
uv run positronic eval status --id=<hex id>
```

`eval run` is the same command that runs an eval on the machine in front of you. A policy image in
place of a policy sends it to the platform, and a policy URL files a plan for the lab rig:

```bash
uv run positronic eval catalog
uv run positronic eval run --policy-url=baseline=wss://baseline.example/ws,candidate=wss://candidate.example/ws --tasks=<task id> --episodes=10 --cap=180
```

Two or more `--policy-url` make one blind sample: the operator is told no policy, and each episode
records which one served it. `--from-file` takes the whole plan as a YAML or JSON file.

An eval names the embodiment it runs on — a task suite belongs to a simulator or to one real robot,
never to both — so the eval is the whole of the choice. The platform owns the list. Read the names
from it; do not copy them from a document:

```bash
uv run positronic/cli/examples/nebius_competition/standings.py
```

That prints the public boards and the eval each one ranks, with no key. With a key, `eval run` with
a name the platform does not offer answers with every eval on offer, with a board or without one.

## From Python

```bash
uv run positronic/cli/examples/walkthrough.py --eval=<name> --policy-image=<reference>
```

`--eval` takes one of the evals `standings.py` prints; with no `--eval`, the walkthrough prints them
and stops. `--policy-image` names an image the platform can pull; there is no public one to default
to. The key comes from `POSITRONIC_PLATFORM_API_KEY`, which `platform-register` prints. A caller who
holds a GitHub token the platform's OAuth app minted sets `POSITRONIC_PLATFORM_CREDENTIAL` instead,
and the walkthrough registers with it. Every script talks to `https://platform.positronic.ro` unless
`--platform-url` says otherwise.

Engagement-specific material lives in its own subdirectory, and
`positronic/cli/tests/test_vocabulary.py` holds everything outside it to that.
