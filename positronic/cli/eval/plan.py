"""The part of `positronic eval run` that files an eval plan for the lab rig."""

from pathlib import Path

import yaml
from platform_client.eval_plan import EvalPlan
from platform_client.responses import SubmissionCreateResponse
from pydantic import ValidationError

from positronic.cli.account.gateway import gateway, one_line

# The plan fields the command line states beside a plan file.
TRANSACTION_KEY_FIELD = 'transaction_key'
ALIAS_FIELD = 'alias'


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


def read_plan(path: Path, transaction_key: str | None = None, alias: str | None = None) -> EvalPlan:
    """The whole plan, from a file. A YAML reader reads JSON too, so one reader takes both forms.

    `--transaction-key` and `--alias` are the plan fields the command line states beside a file: each
    belongs to one filing, and the file names the plan. A file carrying either takes no flag for it.
    """
    try:
        payload = yaml.load(path.read_bytes(), Loader=_OneValuePerKey)  # noqa: S506 — a SafeLoader subclass
    except OSError as exc:
        raise SystemExit(f'{path}: {exc.strerror}') from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f'{path} reads as neither YAML nor JSON: {exc}') from exc
    for field, stated in ((TRANSACTION_KEY_FIELD, transaction_key), (ALIAS_FIELD, alias)):
        if stated is None or not isinstance(payload, dict):
            continue
        if field in payload:
            raise SystemExit(f'{path} carries {field}; drop --{field.replace("_", "-")}')
        payload = {**payload, field: stated}
    try:
        return EvalPlan.model_validate(payload)
    except ValidationError as exc:
        raise SystemExit(f'{path}: {one_line(exc)}') from exc


def given(value: object) -> bool:
    """Whether a flag was given. An unstated flag is `None`, so `0`, `""` and `False` are values."""
    return value is not None


def file_plan(plan: EvalPlan, platform_url: str | None = None) -> SubmissionCreateResponse:
    """File one plan with `submissions.create`, print what came back, and return it.

    `positronic eval status` reads the run back by the id this prints.
    """
    with gateway(platform_url) as client:
        filed = client.create_submission(plan)
    print(filed.model_dump_json(indent=2))
    return filed


def plan_source(eval: object, from_file: str | None) -> Path | None:
    """The file a whole plan comes from. `--from-file` names it, and no other option does.

    An `--eval` value is a name wherever the command runs. Read as a path as well, one token would
    mean two things by what the working directory holds beside it.
    """
    if from_file is None:
        return None
    if eval is not None:
        raise SystemExit(f'--from-file={from_file} carries the whole plan; drop --eval')
    return Path(from_file)
