"""The part of `positronic eval run` that files an eval plan for the lab rig."""

from collections.abc import Mapping
from pathlib import Path

import yaml
from platform_client.eval_plan import EvalPlan
from platform_client.responses import SubmissionCreateResponse
from pydantic import ValidationError

from positronic.cli.account.gateway import gateway, one_line

# The plan field `--transaction-key` states beside a plan file.
TRANSACTION_KEY_FIELD = 'transaction_key'


class _OneValuePerKey(yaml.SafeLoader):
    """`yaml.safe_load` keeps the last of two equal keys. A plan that repeats one states two counts or
    two caps, and the one it keeps is a typo, so a repeated key is refused by name."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, str) and key in seen:
                raise yaml.constructor.ConstructorError(None, None, f'{key!r} is given twice', key_node.start_mark)
            if isinstance(key, str):
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


def read_plan(path: Path, transaction_key: str | None = None) -> EvalPlan:
    """The whole plan, from a file. A YAML reader reads JSON too, so one reader takes both forms.

    `--transaction-key` is the one plan field the command line states beside a file: a key names
    one filing, and the file names the plan. A file that carries its own key takes no flag.
    """
    try:
        payload = yaml.load(path.read_bytes(), Loader=_OneValuePerKey)  # noqa: S506 — a SafeLoader subclass
    except OSError as exc:
        raise SystemExit(f'{path}: {exc.strerror}') from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f'{path} reads as neither YAML nor JSON: {exc}') from exc
    if transaction_key is not None and isinstance(payload, dict):
        if TRANSACTION_KEY_FIELD in payload:
            raise SystemExit(f'{path} carries {TRANSACTION_KEY_FIELD}; drop --transaction-key')
        payload = {**payload, TRANSACTION_KEY_FIELD: transaction_key}
    try:
        return EvalPlan.model_validate(payload)
    except ValidationError as exc:
        raise SystemExit(f'{path}: {one_line(exc)}') from exc


def given(value: object) -> bool:
    """Whether a flag was given. An unstated flag is `None`, so `0`, `""` and `False` are values."""
    return value is not None


def flag_entries(value: object, flag: str) -> list[str]:
    """The entries of a repeatable flag, in both forms the command line produces.

    The CLI reads a value with `ast.literal_eval`: `[a,b]` arrives as a list, and a value it cannot
    read (a hyphen or a `=` inside the brackets) arrives as text. This splits the text on commas,
    so `--tasks=[a,b]` and `--tasks=a-b,c-d` state one list.
    """
    if value is None:
        return []
    if isinstance(value, list | tuple):
        entries = [str(entry) for entry in value]
    elif isinstance(value, str):
        entries = value.removeprefix('[').removesuffix(']').split(',')
    else:
        raise SystemExit(f'{flag} takes text; quote a value that reads as a number: \'"{value}"\'')
    stripped = [entry.strip() for entry in entries]
    if not all(stripped):
        raise SystemExit(f'{flag} carries an empty entry: {value!r}')
    return stripped


def endpoint_of(spec: str, position: int) -> dict[str, str]:
    """One `--policy-url` entry: `NAME=URL`, or a bare URL named for its place in the list.

    A URL carries `=` in a query string, so the part before the first one is a label only where it
    names no scheme and no path.
    """
    label, separator, address = spec.partition('=')
    if separator and ':' not in label and '/' not in label:
        return {'name': label, 'url': address}
    return {'name': f'policy{position}', 'url': spec}


def plan_from_flags(
    *,
    policy_url: object,
    tasks: object,
    episodes: int | None,
    cap: int | None,
    preset: str | None,
    transaction_key: str | None,
) -> EvalPlan:
    """The plan the rig flags state."""
    task_ids = flag_entries(tasks, '--tasks')
    urls = flag_entries(policy_url, '--policy-url')
    if not task_ids or not urls or episodes is None:
        raise SystemExit('a rig run states --tasks, --policy-url and --episodes, or the whole plan in a file')
    payload: dict[str, object] = {
        'tasks': task_ids,
        'endpoints': [endpoint_of(spec, position) for position, spec in enumerate(urls, start=1)],
        'episodes_per_endpoint': episodes,
        'cap_per_episode_sec': cap,
        'policy_preset': preset,
        'transaction_key': transaction_key,
    }
    try:
        return EvalPlan.model_validate(payload)
    except ValidationError as exc:
        raise SystemExit(one_line(exc)) from exc


def file_plan(plan: EvalPlan, platform_url: str | None = None) -> SubmissionCreateResponse:
    """File one plan with `submissions.create`, print what came back, and return it.

    `positronic eval status` reads the run back by the id this prints.
    """
    with gateway(platform_url) as client:
        filed = client.create_submission(plan)
    print(filed.model_dump_json(indent=2))
    return filed


def plan_source(eval: object, from_file: str | None) -> Path | None:
    """The file a whole plan comes from: `--from-file`, else an `--eval` that names an existing file."""
    if from_file is not None:
        if eval is not None:
            raise SystemExit(f'--from-file={from_file} carries the whole plan; drop --eval')
        return Path(from_file)
    return Path(eval) if isinstance(eval, str) and Path(eval).is_file() else None


def refusing_a_second_source(source: Path, stated: Mapping[str, object]) -> None:
    """Exit when a plan file and plan flags are both given: one source states the plan."""
    twice = sorted(flag for flag, value in stated.items() if given(value))
    if twice:
        raise SystemExit(f'{source} carries the whole plan; drop {", ".join(twice)}')
