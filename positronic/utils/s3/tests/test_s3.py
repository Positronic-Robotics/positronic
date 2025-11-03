import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

import positronic.utils.s3 as s3

BOTO3_PATCH_TARGET = 'positronic.utils.s3.boto3.client'


def _make_404_error(*_args, **_kwargs):
    raise ClientError({'Error': {'Code': '404'}}, 'head_object')


def _setup_s3_mock(mock_boto_client, paginate_return_value=None):
    mock_s3 = Mock()
    mock_boto_client.return_value = mock_s3

    mock_s3.head_object.side_effect = _make_404_error

    mock_paginator = Mock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = paginate_return_value or [{'Contents': []}]

    return mock_s3


class TestS3URLParsing:
    def test_parse_s3_url_valid(self):
        assert s3._parse_s3_url('s3://bucket/path/to/data') == ('bucket', 'path/to/data')
        assert s3._parse_s3_url('s3://bucket/') == ('bucket', '')

    def test_parse_s3_url_invalid_scheme(self):
        with pytest.raises(ValueError, match='Not an S3 URL'):
            s3._parse_s3_url('http://bucket/path')


class TestMirrorLifecycle:
    def test_download_requires_active_mirror(self):
        with pytest.raises(RuntimeError, match='No active mirror'):
            s3.download('s3://bucket/data')

    def test_nested_mirror_fails(self):
        with s3.mirror(show_progress=False):
            with pytest.raises(RuntimeError, match='Mirror already active'):
                with s3.mirror():
                    pass


class TestDownload:
    @patch(BOTO3_PATCH_TARGET)
    def test_download_deduplicated(self, mock_boto_client):
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                path1 = s3.download('s3://bucket/data')
                path2 = s3.download('s3://bucket/data')

        assert path1 == path2
        assert mock_s3.download_file.call_count == 1

    @patch(BOTO3_PATCH_TARGET)
    def test_download_local_override_conflict(self, mock_boto_client):
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_a = Path(tmpdir) / 'custom_a'
            custom_b = Path(tmpdir) / 'custom_b'

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=custom_a)
                with pytest.raises(ValueError, match='already registered'):
                    s3.download('s3://bucket/data', local=custom_b)

    def test_download_local_passthrough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / 'data'
            local_path.mkdir()

            with s3.mirror(show_progress=False):
                resolved = s3.download(str(local_path))

        assert resolved == local_path.resolve()

    @patch(BOTO3_PATCH_TARGET)
    def test_thread_safe_download(self, mock_boto_client):
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):

                def _do_download(_):
                    return s3.download('s3://bucket/data')

                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(_do_download, range(4)))

        assert len(set(results)) == 1
        assert mock_s3.download_file.call_count == 1


class TestUpload:
    @patch(BOTO3_PATCH_TARGET)
    def test_upload_conflict_with_download(self, mock_boto_client):
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data')
                with pytest.raises(ValueError, match='Conflict'):
                    s3.upload('s3://bucket/data/subdir')

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_deduplicated(self, mock_boto_client):
        _setup_s3_mock(mock_boto_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                path1 = s3.upload('s3://bucket/output')
                path2 = s3.upload('s3://bucket/output')

        assert path1 == path2

    def test_upload_local_passthrough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / 'output'

            with s3.mirror(show_progress=False):
                resolved = s3.upload(str(local_path))

            assert resolved == local_path.resolve()
            assert local_path.exists()

    @patch(BOTO3_PATCH_TARGET)
    def test_final_sync_upload(self, mock_boto_client):
        paginate = [{'Contents': [{'Key': 'output/existing.txt', 'Size': 5}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'output'
            output.mkdir()
            (output / 'new.txt').write_text('content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.upload('s3://bucket/output', local=output, interval=None)

        assert mock_s3.upload_file.call_count >= 1
        assert mock_s3.delete_object.call_count == 1

    @patch(BOTO3_PATCH_TARGET)
    def test_background_sync_uploads_repeatedly(self, mock_boto_client):
        mock_s3 = _setup_s3_mock(mock_boto_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'output'
            output.mkdir()
            (output / 'data.txt').write_text('content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.upload('s3://bucket/output', local=output, interval=1)
                time.sleep(2.5)

        assert mock_s3.upload_file.call_count >= 2
