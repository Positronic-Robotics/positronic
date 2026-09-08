"""The key record: where a registration saves its key, and the order every command reads one in.

`register` writes `config.json` under the config directory, mode 0600, holding the platform URL and
the key together. A command reads the key from `POSITRONIC_PLATFORM_API_KEY`, then from that
record. It reads the platform from its own argument, then from `POSITRONIC_PLATFORM_URL`, then from
that record.
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

    Unchecked, a blank key reaches the gateway as an empty `Bearer` header, and the failure names no
    file.
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
    """The record `register` wrote under `directory`, or `None` when the file is absent.

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

    `mkstemp` names the staged file, so a symlink planted beside the record is never opened. A
    failed write removes the staged file, so no key is left beside the record.
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


def key_is_given(env: Mapping[str, str]) -> bool:
    """Whether the caller names a key in the environment. A named key wins over the record's."""
    return bool(env.get(API_KEY_ENV))


def api_key_from(env: Mapping[str, str], record: Config | None) -> ApiKey | None:
    """The key to call with: from the environment, then from the record."""
    from_env = env.get(API_KEY_ENV)
    if from_env is not None and not from_env.strip():
        raise SystemExit(f'{API_KEY_ENV} is set to an empty value: put the key in it, or unset it')
    if from_env:
        return ApiKey(from_env)
    return record.api_key if record else None


def platform_is_given(env: Mapping[str, str], platform_url: str | None) -> bool:
    """Whether the caller names a platform, by argument or in the environment.

    A named platform wins over the record's.
    """
    return platform_url is not None or env.get(API_URL_ENV) is not None


def platform_url_from(env: Mapping[str, str], platform_url: str | None, record: Config | None) -> str | None:
    """The platform URL the client resolves: the argument, then the environment, then the record."""
    if platform_url is not None:
        return platform_url
    from_env = env.get(API_URL_ENV)
    if from_env is not None:
        return from_env
    return record.platform_url if record else None


def same_platform(one: str, other: str) -> bool:
    """Whether two URLs name one platform. Both are parsed, and a trailing slash is ignored."""
    return httpx.URL(one.removesuffix('/')) == httpx.URL(other.removesuffix('/'))


def record_if_needed(env: Mapping[str, str], platform_url: str | None) -> Config | None:
    """The saved record, or `None` when the caller names both the key and the platform."""
    if key_is_given(env) and platform_is_given(env, platform_url):
        return None
    return read_config(config_dir(env))
