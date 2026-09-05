"""`positronic-platform` — register, then file and read rollout requests, from a terminal or an agent.

The key is never an argument. `register` writes `config.json` under the config directory, mode
0600, holding the platform URL and the key together; every command reads it. The key is read from
`POSITRONIC_PLATFORM_API_KEY`, else from the file `--api-key-file` names, else from that record; the
platform is `--platform-url`, else `POSITRONIC_PLATFORM_URL`, else that record, else production. The
config directory is `POSITRONIC_PLATFORM_CONFIG_DIR`, else `~/.config/positronic-platform`. Every
command prints its answer as JSON.

Usage
  positronic-platform register --alias='<display name>'
  positronic-platform requests create --tasks eight-spoons-into-grey-tote --endpoints gyros=wss://host/ws \\
      --episodes-per-endpoint 10 --cap 180 --preset runway_ziyi --scene tote_placement=random
  positronic-platform requests create --from request.json          # a RequestCreate, as JSON
  positronic-platform requests get 1f
  positronic-platform requests list --after 1f --limit 50
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from platform_client import github_device_flow
from platform_client.client import API_KEY_ENV, API_URL_ENV, PlatformClient, resolve_base_url
from platform_client.enums import CameraVantage, KeyStatus, Placement
from platform_client.errors import PlatformError
from platform_client.ids import ApiKey, RequestId, TransactionKey, UserId
from platform_client.requests import EndpointAsk, RequestCreate, SceneAsk, TaskAsk
from platform_client.slug import Slugged, slug_of
from platform_client.tasks import TaskRef
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

CONFIG_DIR_ENV = 'POSITRONIC_PLATFORM_CONFIG_DIR'
DEFAULT_CONFIG_DIR = Path('~/.config/positronic-platform')
CONFIG_FILENAME = 'config.json'

# What `--scene` takes: `tote_placement=<side>`, `camera_vantage=<vantage>`, and `camera.<mount>=<side>`.
SCENE_TOTE = 'tote_placement'
SCENE_VANTAGE = 'camera_vantage'
SCENE_CAMERA_PREFIX = 'camera.'

_PLACEMENT: TypeAdapter[Placement] = TypeAdapter(Slugged[Placement])
_VANTAGE: TypeAdapter[CameraVantage] = TypeAdapter(Slugged[CameraVantage])


class Config(BaseModel):
    """What `register` records and every command reads: the platform, and the key that belongs to it."""

    model_config = ConfigDict(extra='forbid')

    platform_url: str
    api_key: ApiKey


def config_dir(env: Mapping[str, str]) -> Path:
    return Path(env.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR).expanduser()


def read_config(directory: Path) -> Config | None:
    """The record `register` wrote under `directory`, or None where there is none.

    A file that is not a record ends the command with one line naming it: the traceback would
    print the file, and the file may hold a key.
    """
    path = directory / CONFIG_FILENAME
    try:
        return Config.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except ValidationError as exc:
        raise SystemExit(f'{path} is not a config record: delete it and run `positronic-platform register`') from exc


def write_config(directory: Path, config: Config) -> None:
    """Record the pair as one file, mode 0600, by rename: a reader sees the previous record or this one.

    The staged file is created for this write alone, under a name of its own, so a path planted
    beside the record is not written through.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / CONFIG_FILENAME
    descriptor, staged = tempfile.mkstemp(dir=directory, prefix=f'.{CONFIG_FILENAME}.')
    with os.fdopen(descriptor, 'w') as staged_file:
        staged_file.write(config.model_dump_json(indent=2))
    os.replace(staged, path)


def key_is_given(env: Mapping[str, str], api_key_file: Path | None) -> bool:
    """Whether the caller names a key of their own — the environment or a key file — over the record's."""
    return bool(env.get(API_KEY_ENV)) or api_key_file is not None


def api_key_from(env: Mapping[str, str], api_key_file: Path | None, record: Config | None) -> ApiKey | None:
    """The key to call with: the environment's, else the named file's, else the record's."""
    from_env = env.get(API_KEY_ENV)
    if from_env:
        return ApiKey(from_env)
    if api_key_file is not None:
        try:
            value = api_key_file.read_text().strip()
        except FileNotFoundError:
            return None
        return ApiKey(value) if value else None
    return record.api_key if record else None


def platform_is_given(env: Mapping[str, str], platform_url: str | None) -> bool:
    """Whether the caller names a platform of their own — the flag or the environment — over the record's."""
    return platform_url is not None or env.get(API_URL_ENV) is not None


def platform_url_from(env: Mapping[str, str], platform_url: str | None, record: Config | None) -> str | None:
    """What the client resolves the platform from: the flag, else the record — unless the environment names one."""
    if platform_is_given(env, platform_url):
        return platform_url
    return record.platform_url if record else None


def record_if_needed(env: Mapping[str, str], api_key_file: Path | None, platform_url: str | None) -> Config | None:
    """The saved record, read only where the caller leaves the key or the platform to it."""
    if key_is_given(env, api_key_file) and platform_is_given(env, platform_url):
        return None
    return read_config(config_dir(env))


def _one_line(exc: ValidationError) -> str:
    return '; '.join(f'{".".join(str(part) for part in error["loc"])}: {error["msg"]}' for error in exc.errors())


def _endpoint(spec: str) -> EndpointAsk:
    """`NAME` or `NAME=URL`, as `--endpoints` takes each entry."""
    name, has_url, url = spec.partition('=')
    return EndpointAsk(name=name, url=url if has_url else None)


def scene_from_pairs(pairs: Sequence[str]) -> SceneAsk | None:
    """`--scene KEY=VALUE` pairs as one `SceneAsk`, or None for none. An unknown key is a `SystemExit`."""
    if not pairs:
        return None
    tote: Placement | None = None
    vantage: CameraVantage | None = None
    cameras: dict[str, Placement] = {}
    for pair in pairs:
        key, has_value, value = pair.partition('=')
        if not has_value:
            raise SystemExit(f'--scene takes KEY=VALUE, not {pair!r}')
        try:
            if key == SCENE_TOTE:
                tote = _PLACEMENT.validate_python(value)
            elif key == SCENE_VANTAGE:
                vantage = _VANTAGE.validate_python(value)
            elif key.startswith(SCENE_CAMERA_PREFIX) and len(key) > len(SCENE_CAMERA_PREFIX):
                cameras[key.removeprefix(SCENE_CAMERA_PREFIX)] = _PLACEMENT.validate_python(value)
            else:
                raise SystemExit(
                    f'--scene takes {SCENE_TOTE}, {SCENE_VANTAGE} or {SCENE_CAMERA_PREFIX}<mount>, not {key!r}'
                )
        except ValidationError as exc:
            raise SystemExit(f'--scene {pair}: {_one_line(exc)}') from exc
    return SceneAsk(tote_placement=tote, camera_vantage=vantage, external_cameras=cameras)


def ask_from_args(args: argparse.Namespace) -> RequestCreate:
    """The request `requests create` files: the whole of `--from`, or one built from the flags."""
    flags = (
        args.tasks,
        args.endpoints,
        args.episodes_per_endpoint,
        args.cap,
        args.preset,
        args.slug,
        args.transaction_key,
    )
    if args.from_file is not None:
        if any(flag is not None for flag in flags) or args.scene:
            raise SystemExit('--from carries the whole request; give it alone')
        try:
            return RequestCreate.model_validate_json(args.from_file.read_bytes())
        except OSError as exc:
            raise SystemExit(f'--from {args.from_file}: {exc.strerror}') from exc
        except ValidationError as exc:
            raise SystemExit(f'--from {args.from_file}: {_one_line(exc)}') from exc
    if not args.tasks or args.episodes_per_endpoint is None:
        raise SystemExit('name --tasks and --episodes-per-endpoint, or give the whole request with --from')
    try:
        return RequestCreate(
            tasks=[TaskAsk(task_id=TaskRef(task_id)) for task_id in args.tasks],
            endpoints=[_endpoint(spec) for spec in args.endpoints or []],
            episodes_per_endpoint=args.episodes_per_endpoint,
            cap_per_episode_sec=args.cap,
            policy_preset=args.preset,
            scene=scene_from_pairs(args.scene),
            slug=args.slug,
            transaction_key=TransactionKey(args.transaction_key) if args.transaction_key is not None else None,
        )
    except ValidationError as exc:
        raise SystemExit(_one_line(exc)) from exc
    except ValueError as exc:  # a `TaskRef` that could never be a catalogue key
        raise SystemExit(str(exc)) from exc


def _client(args: argparse.Namespace, env: Mapping[str, str]) -> PlatformClient:
    """The client to call with. The record's key reaches the record's platform and no other."""
    record = record_if_needed(env, args.api_key_file, args.platform_url)
    api_key = api_key_from(env, args.api_key_file, record)
    if api_key is None:
        raise SystemExit(f'no API key: set {API_KEY_ENV}, pass --api-key-file, or run `positronic-platform register`')
    try:
        base_url = resolve_base_url(platform_url_from(env, args.platform_url, record))
    except (ValueError, httpx.InvalidURL) as exc:
        raise SystemExit(str(exc)) from exc
    if record is not None and not key_is_given(env, args.api_key_file) and base_url != record.platform_url:
        raise SystemExit(
            f'the saved key belongs to {record.platform_url}, and this command names {base_url}: pass '
            f'--api-key-file with a key for that platform, or register there with '
            f'`positronic-platform register --platform-url={base_url}`'
        )
    return PlatformClient(base_url, api_key=api_key)


def _show(model: BaseModel) -> None:
    print(model.model_dump_json(indent=2))


class Registered(BaseModel):
    """What `register` prints: the account, the platform, and where the key went. The key itself stays out."""

    user_id: UserId
    key_status: Slugged[KeyStatus]
    platform_url: str
    # The record holding the key this run minted, or None where none was issued: a repeat
    # registration mints none, and the record already on disk stays as it is.
    config_file: Path | None = None


def _register(args: argparse.Namespace, env: Mapping[str, str]) -> None:
    record = None if platform_is_given(env, args.platform_url) else read_config(config_dir(env))
    base_url = github_device_flow.allowed_platform_url(
        platform_url_from(env, args.platform_url, record), plaintext_http=args.plaintext_http
    )
    response = github_device_flow.run_registration(args.client_id, base_url, alias=args.alias, rotate=args.rotate)
    config_file: Path | None = None
    if response.api_key is not None:
        config_file = config_dir(env) / CONFIG_FILENAME
        try:
            write_config(config_dir(env), Config(platform_url=base_url, api_key=response.api_key))
        except OSError as exc:
            # The key is out and cannot be read back, so the message says what to do and never shows it.
            raise SystemExit(
                f'the platform issued a key for user {response.user_id}, and writing {config_file} failed: '
                f'{exc.strerror or exc}. The key is not shown, and the platform refuses a rotation today: '
                'contact the operator to reset the account, then run `positronic-platform register` again.'
            ) from exc
    _show(
        Registered(
            user_id=response.user_id, key_status=response.key_status, platform_url=base_url, config_file=config_file
        )
    )


def _create(args: argparse.Namespace, env: Mapping[str, str]) -> None:
    ask = ask_from_args(args)
    with _client(args, env) as client:
        _show(client.requests_create(ask))


def _request_id(raw: str) -> RequestId:
    try:
        return RequestId.parse(raw)
    except ValueError as exc:
        raise SystemExit(f'not a request id: {raw!r} (the hex id `requests create` printed)') from exc


def _get(args: argparse.Namespace, env: Mapping[str, str]) -> None:
    with _client(args, env) as client:
        _show(client.requests_get(_request_id(args.id)))


def _list(args: argparse.Namespace, env: Mapping[str, str]) -> None:
    after = _request_id(args.after) if args.after is not None else None
    with _client(args, env) as client:
        _show(client.requests_list(after=after, limit=args.limit))


def _refusal(exc: PlatformError) -> str:
    """The refusal as one line, with the task catalogue when the platform sent it back."""
    line = f'{slug_of(exc.code)}: {exc.message}'
    tasks = exc.tasks
    if tasks is not None:
        line += f'\nthe catalogue holds: {", ".join(tasks)}'
    return line


def _positive_int(text: str) -> int:
    """An argparse type for a count: `--limit 0` ends the command at the parser, with one line."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'{text} is not a positive integer')
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='positronic-platform', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest='command', required=True)

    register = commands.add_parser('register', help='mint the key every other command needs, through GitHub')
    github_device_flow.add_arguments(register)
    register.set_defaults(run=_register)

    calling = argparse.ArgumentParser(add_help=False)
    calling.add_argument('--platform-url', default=None, help=f'the platform to call, else {API_URL_ENV}')
    calling.add_argument(
        '--api-key-file', type=Path, default=None, help=f'a file holding the key alone, else {API_KEY_ENV}'
    )

    requests = commands.add_parser('requests', help='file and read rollout requests')
    verbs = requests.add_subparsers(dest='verb', required=True)

    create = verbs.add_parser('create', parents=[calling], help='file one request')
    create.add_argument('--tasks', nargs='+', default=None, metavar='TASK_ID', help='the catalogue tasks to run')
    create.add_argument(
        '--endpoints', nargs='+', default=None, metavar='NAME[=URL]', help='the policies to run, by label'
    )
    create.add_argument(
        '--episodes-per-endpoint', type=int, default=None, help='episodes each endpoint of each task takes'
    )
    create.add_argument('--cap', type=int, default=None, help='the window one episode is given, in seconds')
    create.add_argument('--preset', default=None, help='the policy preset a blind run is built from')
    create.add_argument(
        '--scene',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help=f'{SCENE_TOTE}=<side>, {SCENE_VANTAGE}=<vantage>, or {SCENE_CAMERA_PREFIX}<mount>=<side>',
    )
    create.add_argument('--slug', default=None, help='a short name for the request')
    create.add_argument('--transaction-key', default=None, help='a key that makes a retry return the same request')
    create.add_argument(
        '--from', dest='from_file', type=Path, default=None, metavar='FILE', help='the whole request, as JSON'
    )
    create.set_defaults(run=_create)

    get = verbs.add_parser('get', parents=[calling], help='one request, by id')
    get.add_argument('id', help='the hex id `requests create` printed')
    get.set_defaults(run=_get)

    listing = verbs.add_parser('list', parents=[calling], help="the caller's requests, oldest first")
    listing.add_argument('--after', default=None, metavar='ID', help='the last id seen: the page after it')
    listing.add_argument('--limit', type=_positive_int, default=None, help='rows per page, a positive integer')
    listing.set_defaults(run=_list)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.run(args, os.environ)
    except PlatformError as exc:
        raise SystemExit(_refusal(exc)) from exc
    except httpx.HTTPError as exc:
        raise SystemExit(f'the platform did not answer: {exc}') from exc


if __name__ == '__main__':
    main()
