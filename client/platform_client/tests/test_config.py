"""The key record: what `write_config` puts on disk, and the order a command reads a key in."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from platform_client.client import API_KEY_ENV, API_URL_ENV
from platform_client.config import (
    CONFIG_DIR_ENV,
    CONFIG_FILENAME,
    Config,
    api_key_from,
    config_dir,
    key_is_given,
    platform_is_given,
    platform_url_from,
    read_config,
    record_if_needed,
    same_platform,
    write_config,
)
from platform_client.ids import ApiKey
from pydantic import ValidationError

PLATFORM = 'https://platform.test'
KEY = ApiKey('pk_live_secret')


@pytest.fixture
def record(tmp_path: Path) -> Config:
    saved = Config(platform_url=PLATFORM, api_key=KEY)
    write_config(tmp_path, saved)
    return saved


def test_a_written_record_reads_back_whole(tmp_path: Path, record: Config):
    assert read_config(tmp_path) == record


def test_a_directory_with_no_record_reads_as_none(tmp_path: Path):
    assert read_config(tmp_path) is None


def test_the_record_is_written_under_a_directory_only_its_owner_can_read(tmp_path: Path):
    directory = tmp_path / 'fresh'
    write_config(directory, Config(platform_url=PLATFORM, api_key=KEY))
    assert directory.stat().st_mode & 0o077 == 0
    assert (directory / CONFIG_FILENAME).exists()


def test_a_torn_write_leaves_the_previous_record_whole(tmp_path: Path, record: Config, monkeypatch):
    def fails(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(os, 'replace', fails)
    with pytest.raises(OSError, match='disk full'):
        write_config(tmp_path, Config(platform_url='https://other.test', api_key=ApiKey('second')))

    assert read_config(tmp_path) == record
    # The staged file carries the key too, so a failed write leaves none of it beside the record.
    assert [path.name for path in tmp_path.iterdir()] == [CONFIG_FILENAME]


def test_a_path_planted_beside_the_record_is_not_written_through(tmp_path: Path):
    # `mkstemp` names the staged file itself, so a symlink an attacker leaves here is never opened.
    (tmp_path / f'.{CONFIG_FILENAME}.planted').symlink_to(tmp_path / 'elsewhere')
    write_config(tmp_path, Config(platform_url=PLATFORM, api_key=KEY))

    assert not (tmp_path / 'elsewhere').exists()
    assert read_config(tmp_path) is not None


def test_a_file_that_is_not_a_record_names_itself_and_shows_nothing_of_it(tmp_path: Path):
    (tmp_path / CONFIG_FILENAME).write_text('{"api_key": "leaked-key"}')

    with pytest.raises(SystemExit) as exit_info:
        read_config(tmp_path)

    assert CONFIG_FILENAME in str(exit_info.value)
    assert 'leaked-key' not in str(exit_info.value)


@pytest.mark.parametrize('blank', ['', '   '])
def test_a_record_holding_a_blank_key_is_a_malformed_record(blank: str):
    with pytest.raises(ValidationError, match='api_key is blank'):
        Config(platform_url=PLATFORM, api_key=ApiKey(blank))


@pytest.mark.parametrize('bad', ['', '/v1', 'http://host:bad'])
def test_a_record_whose_platform_names_no_host_is_a_malformed_record(bad: str):
    with pytest.raises(ValidationError):
        Config(platform_url=bad, api_key=KEY)


def test_the_config_directory_comes_from_the_environment(tmp_path: Path):
    assert config_dir({CONFIG_DIR_ENV: str(tmp_path)}) == tmp_path
    assert config_dir({}).is_absolute()


def test_a_config_directory_set_to_nothing_is_refused_rather_than_ignored():
    with pytest.raises(SystemExit, match=CONFIG_DIR_ENV):
        config_dir({CONFIG_DIR_ENV: ''})


def test_the_environment_holds_the_key_before_a_file_and_a_file_before_the_record(tmp_path: Path, record: Config):
    key_file = tmp_path / 'key'
    key_file.write_text('from-the-file\n')

    assert api_key_from({API_KEY_ENV: 'from-the-environment'}, key_file, record) == 'from-the-environment'
    assert api_key_from({}, key_file, record) == 'from-the-file'
    assert api_key_from({}, None, record) == KEY
    assert api_key_from({}, None, None) is None


def test_a_key_variable_set_to_nothing_is_refused_rather_than_ignored(record: Config):
    with pytest.raises(SystemExit, match=API_KEY_ENV):
        api_key_from({API_KEY_ENV: '  '}, None, record)


def test_a_key_file_that_is_missing_or_empty_ends_the_command_naming_it(tmp_path: Path):
    empty = tmp_path / 'empty'
    empty.write_text('  \n')
    with pytest.raises(SystemExit, match='holds no key'):
        api_key_from({}, empty, None)
    with pytest.raises(SystemExit, match='absent'):
        api_key_from({}, tmp_path / 'absent', None)


def test_a_key_file_that_is_not_utf8_names_it_and_quotes_none_of_it(tmp_path: Path):
    key_file = tmp_path / 'key'
    key_file.write_bytes(b'\xff\xfe secret')

    with pytest.raises(SystemExit) as exit_info:
        api_key_from({}, key_file, None)

    assert str(exit_info.value) == f'--api-key-file {key_file} is not UTF-8 text'


def test_the_platform_comes_from_the_argument_then_the_environment_then_the_record(record: Config):
    assert platform_url_from({API_URL_ENV: 'http://env.test'}, 'http://arg.test', record) == 'http://arg.test'
    assert platform_url_from({API_URL_ENV: 'http://env.test'}, None, record) == 'http://env.test'
    assert platform_url_from({}, None, record) == PLATFORM
    assert platform_url_from({}, None, None) is None


def test_an_empty_platform_in_the_environment_is_carried_rather_than_read_as_unset(record: Config):
    # The client refuses it, naming the variable; falling through to the record would send the call
    # to a platform the caller did not name.
    assert platform_url_from({API_URL_ENV: ''}, None, record) == ''


def test_a_caller_names_a_key_or_a_platform_of_their_own(tmp_path: Path):
    assert key_is_given({API_KEY_ENV: 'k'}, None)
    assert key_is_given({}, tmp_path / 'key')
    assert not key_is_given({API_KEY_ENV: ''}, None)
    assert platform_is_given({}, 'http://arg.test')
    assert platform_is_given({API_URL_ENV: 'http://env.test'}, None)
    assert not platform_is_given({}, None)


def test_a_record_a_command_needs_nothing_from_is_not_read(tmp_path: Path, record: Config):
    env = {CONFIG_DIR_ENV: str(tmp_path), API_KEY_ENV: 'k', API_URL_ENV: 'http://env.test'}
    assert record_if_needed(env, None, None) is None
    assert record_if_needed({CONFIG_DIR_ENV: str(tmp_path)}, None, None) == record


def test_a_trailing_slash_names_the_same_platform_and_another_path_does_not():
    assert same_platform('https://platform.test/', 'https://platform.test')
    assert not same_platform('https://platform.test', 'https://other.test')
    assert not same_platform('https://platform.test/v1', 'https://platform.test')
