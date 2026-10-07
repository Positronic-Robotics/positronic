"""The part of `positronic eval run` that files an eval plan for the lab rig."""

import functools
from pathlib import Path

import yaml
from platform_client.eval_plan import EvalPlan, PrivateEval, RegistryCredentialFile, plan_with_passwords_read
from platform_client.ids import OrgSlug
from platform_client.responses import SubmissionCreateResponse
from pydantic import ValidationError

from positronic.cli.account.gateway import gateway, one_line

# The plan fields the command line states beside a plan file.
TRANSACTION_KEY_FIELD = 'transaction_key'
ALIAS_FIELD = 'alias'
REQUEST_TYPE_FIELD = 'request_type'


class _KeyGivenTwice(yaml.constructor.ConstructorError):
    """A plan repeated a mapping key. The message names the key only where it is a plan field."""


@functools.cache
def _plan_field_names() -> frozenset[str]:
    """Every name a plan file's schema declares, at any depth."""
    schema = EvalPlan[RegistryCredentialFile].model_json_schema()
    names = set(schema.get('properties', ()))
    for definition in schema.get('$defs', {}).values():
        names.update(definition.get('properties', ()))
    return frozenset(names)


class _OneValuePerKey(yaml.SafeLoader):
    """`yaml.safe_load` keeps the last of two equal keys. A plan that repeats one states two counts or
    two caps, and the one it keeps is a typo, so a repeated key is refused.

    This runs over every mapping in the file, so a key is not always a plan field. A key the plan
    does not declare is a caller's own text, and goes unnamed.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, str) and key in seen:
                stated = repr(key) if key in _plan_field_names() else 'a key'
                raise _KeyGivenTwice(None, None, f'{stated} is given twice', key_node.start_mark)
            if isinstance(key, str):
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _refusal(exc: yaml.YAMLError) -> str:
    """Why the plan was refused, and where, in this file's own words.

    PyYAML renders what it read into the mark's snippet and into `problem`: a source line, an
    alias, an anchor, a tag. A refusal prints none of that text, and says where the parser stopped.
    `_KeyGivenTwice` carries this file's own message, whose key `_plan_field_names` has already
    cleared.
    """
    mark = None
    if isinstance(exc, yaml.MarkedYAMLError):
        mark = exc.problem_mark or exc.context_mark
    where = f', at line {mark.line + 1}, column {mark.column + 1}' if mark is not None else ''
    if isinstance(exc, _KeyGivenTwice):
        return f'{exc.problem}{where}'
    return f'it reads as neither YAML nor JSON{where}'


def read_plan(
    path: Path, transaction_key: str | None = None, alias: str | None = None, org: str | None = None
) -> EvalPlan:
    """The whole plan, from a file. A YAML reader reads JSON too, so one reader takes both forms.

    A credential in the file names the file its password is in, and the plan this returns holds the
    password read from it.

    `--transaction-key`, `--alias` and `--org` are the plan fields the command line states beside a
    file. A file carrying one takes no flag for it. `--org` states a private request for that org.
    """
    try:
        payload = yaml.load(path.read_bytes(), Loader=_OneValuePerKey)  # noqa: S506 — a SafeLoader subclass
    except OSError as exc:
        raise SystemExit(f'{path}: {exc.strerror}') from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f'{path}: {_refusal(exc)}') from exc
    # Unvalidated here: `EvalPlan` validates it with the rest of the plan, inside the refusal below.
    request_type = PrivateEval.model_construct(org=OrgSlug(org)).model_dump() if org is not None else None
    stated_fields = ((TRANSACTION_KEY_FIELD, transaction_key), (ALIAS_FIELD, alias), (REQUEST_TYPE_FIELD, request_type))
    for field, stated in stated_fields:
        if stated is None or not isinstance(payload, dict):
            continue
        if field in payload:
            flag = 'org' if field == REQUEST_TYPE_FIELD else field.replace('_', '-')
            raise SystemExit(f'{path} carries {field}; drop --{flag}')
        payload = {**payload, field: stated}
    try:
        return plan_with_passwords_read(EvalPlan[RegistryCredentialFile].model_validate(payload))
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
