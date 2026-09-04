"""`positronic-platform` — register, then file and read rollout requests, from a terminal or an agent.

The key is never an argument. It is read from `POSITRONIC_PLATFORM_API_KEY`, else from the file
`--api-key-file` names, else from `api_key` under the config directory, which `register` writes.
The platform is `--platform-url`, else `POSITRONIC_PLATFORM_URL`, else `platform_url` under the
config directory, else production. The config directory is `POSITRONIC_PLATFORM_CONFIG_DIR`, else
`~/.config/positronic-platform`. Every command prints its answer as JSON.

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
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from platform_client import github_device_flow
from platform_client.client import API_KEY_ENV, API_URL_ENV, PlatformClient
from platform_client.enums import CameraVantage, KeyStatus, Placement
from platform_client.errors import PlatformError
from platform_client.ids import ApiKey, RequestId, TransactionKey, UserId
from platform_client.requests import EndpointAsk, RequestCreate, SceneAsk, TaskAsk
from platform_client.slug import Slugged, slug_of
from pydantic import BaseModel, TypeAdapter, ValidationError

CONFIG_DIR_ENV = 'POSITRONIC_PLATFORM_CONFIG_DIR'
DEFAULT_CONFIG_DIR = Path('~/.config/positronic-platform')
API_KEY_FILENAME = 'api_key'
PLATFORM_URL_FILENAME = 'platform_url'

# What `--scene` takes: `tote_placement=<side>`, `camera_vantage=<vantage>`, and `camera.<mount>=<side>`.
SCENE_TOTE = 'tote_placement'
SCENE_VANTAGE = 'camera_vantage'
SCENE_CAMERA_PREFIX = 'camera.'

_PLACEMENT: TypeAdapter[Placement] = TypeAdapter(Slugged[Placement])
_VANTAGE: TypeAdapter[CameraVantage] = TypeAdapter(Slugged[CameraVantage])


def config_dir(env: Mapping[str, str]) -> Path:
    return Path(env.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR).expanduser()


def _read_line(path: Path) -> str | None:
    """The file's content, stripped, or None where there is no such file."""
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def api_key_from(env: Mapping[str, str], api_key_file: str | None) -> ApiKey | None:
    """The key to call with: the environment's, else the named file's, else the config directory's."""
    from_env = env.get(API_KEY_ENV)
    if from_env:
        return ApiKey(from_env)
    path = Path(api_key_file) if api_key_file else config_dir(env) / API_KEY_FILENAME
    value = _read_line(path)
    return ApiKey(value) if value else None


def platform_url_from(env: Mapping[str, str], platform_url: str | None) -> str | None:
    """What the client resolves the platform from: the flag, else the file — unless the environment names one."""
    if platform_url is not None or env.get(API_URL_ENV) is not None:
        return platform_url
    return _read_line(config_dir(env) / PLATFORM_URL_FILENAME)


def _replace(path: Path, content: str, mode: int) -> None:
    """Write `content` beside `path` and rename it into place, so a reader sees the old file or the new one."""
    staged = path.with_name(f'.{path.name}.{os.getpid()}')
    with os.fdopen(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), 'w') as staged_file:
        staged_file.write(content)
    os.replace(staged, path)


def write_config(directory: Path, *, api_key: ApiKey, platform_url: str) -> None:
    """Record a key and the platform it belongs to, as one pair. The key file is mode 0600.

    The key lands first: a key the platform minted and this side did not write is lost, while a
    platform file that did not land leaves the old platform beside a new key, which the next call
    reports as `unauthorized` and `register --rotate` repairs.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _replace(directory / API_KEY_FILENAME, f'{api_key}\n', 0o600)
    _replace(directory / PLATFORM_URL_FILENAME, f'{platform_url}\n', 0o644)


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
            return RequestCreate.model_validate_json(Path(args.from_file).read_bytes())
        except OSError as exc:
            raise SystemExit(f'--from {args.from_file}: {exc.strerror}') from exc
        except ValidationError as exc:
            raise SystemExit(f'--from {args.from_file}: {_one_line(exc)}') from exc
    if not args.tasks or args.episodes_per_endpoint is None:
        raise SystemExit('name --tasks and --episodes-per-endpoint, or give the whole request with --from')
    try:
        return RequestCreate(
            tasks=[TaskAsk(task_id=task_id) for task_id in args.tasks],
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


def _client(args: argparse.Namespace, env: Mapping[str, str]) -> PlatformClient:
    api_key = api_key_from(env, args.api_key_file)
    if api_key is None:
        raise SystemExit(f'no API key: set {API_KEY_ENV}, pass --api-key-file, or run `positronic-platform register`')
    try:
        return PlatformClient(platform_url_from(env, args.platform_url), api_key=api_key)
    except (ValueError, httpx.InvalidURL) as exc:
        raise SystemExit(str(exc)) from exc


def _show(model: BaseModel) -> None:
    print(model.model_dump_json(indent=2))


class Registered(BaseModel):
    """What `register` prints: the account, the platform, and where the key went. The key itself stays out."""

    user_id: UserId
    key_status: Slugged[KeyStatus]
    platform_url: str
    # The file holding the key this run minted, or None where none was issued: a repeat registration
    # mints none, and the pair already on disk stays as it is.
    api_key_file: Path | None = None


def _register(args: argparse.Namespace, env: Mapping[str, str]) -> None:
    base_url = github_device_flow.allowed_platform_url(
        platform_url_from(env, args.platform_url), plaintext_http=args.plaintext_http
    )
    response = github_device_flow.run_registration(args.client_id, base_url, alias=args.alias, rotate=args.rotate)
    key_path: Path | None = None
    if response.api_key is not None:
        # The key and the platform are one pair: written together, or not at all.
        key_path = config_dir(env) / API_KEY_FILENAME
        write_config(config_dir(env), api_key=response.api_key, platform_url=base_url)
    _show(
        Registered(
            user_id=response.user_id, key_status=response.key_status, platform_url=base_url, api_key_file=key_path
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
    after = _request_id(args.after) if args.after else None
    with _client(args, env) as client:
        _show(client.requests_list(after=after, limit=args.limit))


def _refusal(exc: PlatformError) -> str:
    """The refusal as one line, with the task catalogue when the platform sent it back."""
    line = f'{slug_of(exc.code)}: {exc.message}'
    tasks = exc.tasks
    if tasks is not None:
        line += f'\nthe catalogue holds: {", ".join(tasks)}'
    return line


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
    calling.add_argument('--api-key-file', default=None, help=f'the file holding the key, else {API_KEY_ENV}')

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
    create.add_argument('--from', dest='from_file', default=None, metavar='FILE', help='the whole request, as JSON')
    create.set_defaults(run=_create)

    get = verbs.add_parser('get', parents=[calling], help='one request, by id')
    get.add_argument('id', help='the hex id `requests create` printed')
    get.set_defaults(run=_get)

    listing = verbs.add_parser('list', parents=[calling], help="the caller's requests, oldest first")
    listing.add_argument('--after', default=None, metavar='ID', help='the last id seen: the page after it')
    listing.add_argument('--limit', type=int, default=None, help='rows per page')
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
