# Examples

Runnable walkthroughs of the evaluation platform. Everything here runs through `uv` from a
checkout, so nothing has to be installed first — `uv run` builds the environment it needs.

| Script | What it shows |
|---|---|
| `walkthrough.py` | The whole flow through `PlatformClient`: register, submit, read quota, poll to a terminal status. |
| `nebius_competition/submit_sample.py` | Submitting to one engagement's eval with a transaction key, and waiting for what it scored. |

## From the command line

The same flow, with no Python of your own:

```bash
export POSITRONIC_PLATFORM_CREDENTIAL=<the identity to register with>
uv run positronic account register --alias=<display name>
export POSITRONIC_PLATFORM_API_KEY=<the key printed above>

uv run positronic eval run --eval=<name> --policy-image=org/policy@sha256:…
uv run positronic eval status --submission-id=<hex id>
```

`eval run` is the same command that runs an eval on the machine in front of you; a policy image in
place of a policy is what sends it to the platform.

An eval names the embodiment it runs on — a task suite belongs to a simulator or to one real robot,
never to both — so the eval is the whole of the choice. The platform owns the list, and naming one
it does not offer answers with the ones it does.

## From Python

```bash
POSITRONIC_PLATFORM_CREDENTIAL=<token> uv run positronic/cli/examples/walkthrough.py
```

Both scripts talk to `https://platform.positronic.ro` unless `--platform-url` says otherwise; the
walkthrough defaults to the `fake.smoke` eval, which a platform started with no cloud and no GPU
still runs.

Engagement-specific material lives in its own subdirectory, and
`positronic/cli/tests/test_vocabulary.py` holds everything outside it to that.
