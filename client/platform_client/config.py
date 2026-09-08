"""The key record: where a registration saves its key, and the order every command reads one in.

`register` writes `config.json` under the config directory, mode 0600, holding the platform URL and
the key together. A command reads the key from `POSITRONIC_PLATFORM_API_KEY`, else from the file
`--api-key-file` names, else from that record; and the platform from its own argument, else
`POSITRONIC_PLATFORM_URL`, else that record.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import httpx
from platform_client.client import API_KEY_ENV, API_URL_ENV, require_absolute_url
from platform_client.ids import ApiKey
from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

CONFIG_DIR_ENV = 'POSITRONIC_PLATFORM_CONFIG_DIR'
DEFAULT_CONFIG_DIR = Path('~/.config/positronic-platform')
CONFIG_FILENAME = 'config.json'

# The command that writes the record, named in every message that asks for one.
REGISTER_COMMAND = 'positronic account register'


def checked_api_key(value: str) -> str:
    """The record's key, or a `ValueError` that `read_config` reports as a malformed record.

    A blank one reaches the gateway as an unusable `Bearer` header, so the command ends on a
    transport or authorization failure naming no file. The environment and `--api-key-file` paths
    refuse a blank key, and the record is the third way in.
    """
    if not value.strip():
        raise ValueError('api_key is blank; register again, or put the key in the record')
    return value


def checked_platform_url(value: str) -> str:
    """The record's platform, or a `ValueError` that `read_config` reports as a malformed record.

    `same_platform` and the client both parse this value, so a record holding `http://host:bad`
    would otherwise end the command with an `httpx.InvalidURL` traceback naming no file.
    """
    try:
        require_absolute_url(value, 'platform_url')
    except httpx.InvalidURL as exc:
        raise ValueError(f'platform_url does not parse as a URL: {exc}') from exc
    return value


class Config(BaseModel):
    """What `register` records and every command reads: the platform, and the key that belongs to it."""

    model_config = ConfigDict(extra='forbid')

    platform_url: Annotated[str, AfterValidator(checked_platform_url)]
    api_key: Annotated[ApiKey, AfterValidator(checked_api_key)]


def config_dir(env: Mapping[str, str]) -> Path:
    named = env.get(CONFIG_DIR_ENV)
    if named is not None and not named.strip():
        raise SystemExit(f'{CONFIG_DIR_ENV} is set to an empty value: name a directory, or unset it')
    return Path(named or DEFAULT_CONFIG_DIR).expanduser()


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
    except OSError as exc:
        raise SystemExit(f'{path} cannot be read: {exc.strerror}') from exc
    except ValidationError as exc:
        raise SystemExit(f'{path} is not a config record: delete it and run `{REGISTER_COMMAND}`') from exc


def write_config(directory: Path, config: Config) -> None:
    """Record the pair as one file, mode 0600, by rename: a reader sees the previous record or this one.

    The staged file is created for this write alone, under a name of its own, so a path planted
    beside the record is not written through; a write that fails removes it, so no key stays
    beside the record.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / CONFIG_FILENAME
    descriptor, staged = tempfile.mkstemp(dir=directory, prefix=f'.{CONFIG_FILENAME}.')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as staged_file:
            staged_file.write(config.model_dump_json(indent=2))
        os.replace(staged, path)
    except BaseException:
        os.unlink(staged)
        raise


def key_is_given(env: Mapping[str, str], api_key_file: Path | None) -> bool:
    """Whether the caller names a key of their own — the environment or a key file — over the record's."""
    return bool(env.get(API_KEY_ENV)) or api_key_file is not None


def api_key_from(env: Mapping[str, str], api_key_file: Path | None, record: Config | None) -> ApiKey | None:
    """The key to call with: the environment's, else the named file's, else the record's."""
    from_env = env.get(API_KEY_ENV)
    if from_env is not None and not from_env.strip():
        raise SystemExit(f'{API_KEY_ENV} is set to an empty value: put the key in it, or unset it')
    if from_env:
        return ApiKey(from_env)
    if api_key_file is not None:
        try:
            value = api_key_file.read_text(encoding='utf-8').strip()
        except OSError as exc:
            raise SystemExit(f'--api-key-file {api_key_file}: {exc.strerror}') from exc
        except UnicodeDecodeError as exc:
            # The decoder's own message quotes the offending bytes, which are part of the key.
            raise SystemExit(f'--api-key-file {api_key_file} is not UTF-8 text') from exc
        if not value:
            raise SystemExit(f'--api-key-file {api_key_file} holds no key')
        return ApiKey(value)
    return record.api_key if record else None


def platform_is_given(env: Mapping[str, str], platform_url: str | None) -> bool:
    """Whether the caller names a platform of their own — the argument or the environment — over the record's."""
    return platform_url is not None or env.get(API_URL_ENV) is not None


def platform_url_from(env: Mapping[str, str], platform_url: str | None, record: Config | None) -> str | None:
    """What the client resolves the platform from: the argument, else `env`, else the record."""
    if platform_url is not None:
        return platform_url
    from_env = env.get(API_URL_ENV)
    if from_env is not None:
        return from_env
    return record.platform_url if record else None


def same_platform(one: str, other: str) -> bool:
    """Whether two URLs name one platform: parsed, and read without a trailing slash."""
    return httpx.URL(one.removesuffix('/')) == httpx.URL(other.removesuffix('/'))


def record_if_needed(env: Mapping[str, str], api_key_file: Path | None, platform_url: str | None) -> Config | None:
    """The saved record, read only where the caller leaves the key or the platform to it."""
    if key_is_given(env, api_key_file) and platform_is_given(env, platform_url):
        return None
    return read_config(config_dir(env))
