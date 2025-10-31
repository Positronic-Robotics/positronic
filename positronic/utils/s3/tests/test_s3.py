import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from positronic.utils.s3 import Download, Mirror, Options, Upload, _parse_s3_url

# Update patch target since boto3 is now in __init__ module
BOTO3_PATCH_TARGET = 'positronic.utils.s3.boto3.client'


def _make_404_error(*args, **kwargs):
    """Raise a ClientError for 404 (file not found)"""
    raise ClientError({'Error': {'Code': '404'}}, 'head_object')


def _setup_s3_mock(mock_boto_client, paginate_return_value=None):
    """
    Setup common S3 mocking.

    Args:
        mock_boto_client: The patched boto3.client
        paginate_return_value: Optional return value for paginator.paginate()

    Returns:
        mock_s3: The configured S3 client mock
    """
    mock_s3 = Mock()
    mock_boto_client.return_value = mock_s3

    # Mock head_object to return 404 (not a single file)
    mock_s3.head_object.side_effect = _make_404_error

    # Setup paginator
    if paginate_return_value is not None:
        mock_paginator = Mock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = paginate_return_value

    return mock_s3


class TestS3URLParsing:
    """Test S3 URL parsing functionality"""

    def test_parse_s3_url_valid(self):
        """Test parsing valid S3 URLs"""
        assert _parse_s3_url('s3://my-bucket/path/to/data') == ('my-bucket', 'path/to/data')
        assert _parse_s3_url('s3://my-bucket/path/to/data/') == ('my-bucket', 'path/to/data/')
        assert _parse_s3_url('s3://my-bucket/') == ('my-bucket', '')

    def test_parse_s3_url_invalid_scheme(self):
        """Test parsing URL with invalid scheme"""
        with pytest.raises(ValueError, match='Not an S3 URL'):
            _parse_s3_url('http://my-bucket/path')

    def test_generate_cache_path(self):
        """Test cache path generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            options = Options(cache_root=tmpdir)

            cache_path = options._generate_cache_path('s3://my-bucket/datasets/foo')
            expected = Path(tmpdir) / 'my-bucket' / 'datasets' / 'foo'
            assert cache_path == expected


class TestDownload:
    """Test Download functionality"""

    @patch(BOTO3_PATCH_TARGET)
    def test_download_all_new_files(self, mock_boto_client):
        """Test downloading new files"""
        mock_s3 = _setup_s3_mock(
            mock_boto_client,
            [{'Contents': [{'Key': 'data/file1.txt', 'Size': 100}, {'Key': 'data/file2.txt', 'Size': 200}]}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            options = Options(cache_root=tmpdir, show_progress=False)
            with Mirror(options, dataset=Download('s3://bucket/data')) as m:
                # Verify download_file was called for each file
                assert mock_s3.download_file.call_count == 2

                # Verify local path is returned
                assert m.dataset.exists()

    @patch(BOTO3_PATCH_TARGET)
    def test_download_skips_existing_files(self, mock_boto_client):
        """Test that existing files with matching size are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing file
            local_dir = Path(tmpdir) / 'bucket' / 'data'
            local_dir.mkdir(parents=True)
            existing_file = local_dir / 'file1.txt'
            existing_file.write_bytes(b'x' * 100)

            mock_s3 = _setup_s3_mock(mock_boto_client, [{'Contents': [{'Key': 'data/file1.txt', 'Size': 100}]}])

            options = Options(cache_root=tmpdir, show_progress=False)
            with Mirror(options, dataset=Download('s3://bucket/data')):
                # Should not download since file exists with matching size
                mock_s3.download_file.assert_not_called()

    @patch(BOTO3_PATCH_TARGET)
    def test_download_custom_local_path(self, mock_boto_client):
        """Test downloading to custom local path"""
        _setup_s3_mock(mock_boto_client, [{'Contents': [{'Key': 'data/file1.txt', 'Size': 100}]}])

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / 'custom'
            options = Options(show_progress=False)
            with Mirror(options, dataset=Download('s3://bucket/data', local=str(custom_path))) as m:
                # Verify custom path is used
                assert m.dataset == custom_path


class TestUpload:
    """Test Upload functionality"""

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_new_files(self, mock_boto_client):
        """Test uploading new files"""
        mock_s3 = _setup_s3_mock(mock_boto_client, [{'Contents': []}])

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_text('content1')
            (local_dir / 'file2.txt').write_text('content2')

            options = Options(show_progress=False)
            with Mirror(options, output=Upload('s3://bucket/output', local=str(local_dir), interval=None)):
                pass  # Exit triggers final sync

            # Verify files were uploaded
            assert mock_s3.upload_file.call_count == 2

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_with_deletion(self, mock_boto_client):
        """Test upload with deletion of S3 files not present locally"""
        mock_s3 = _setup_s3_mock(
            mock_boto_client,
            [{'Contents': [{'Key': 'output/old_file.txt', 'Size': 100}, {'Key': 'output/file1.txt', 'Size': 50}]}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_bytes(b'x' * 50)  # Exists on S3
            (local_dir / 'file2.txt').write_text('new')  # New file

            options = Options(show_progress=False)
            with Mirror(options, output=Upload('s3://bucket/output', local=str(local_dir), interval=None, delete=True)):
                pass

            # Should upload file2.txt (new)
            assert mock_s3.upload_file.call_count >= 1

            # Should delete old_file.txt from S3
            assert mock_s3.delete_object.call_count == 1
            delete_call = mock_s3.delete_object.call_args
            assert delete_call[1]['Key'] == 'output/old_file.txt'

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_without_deletion(self, mock_boto_client):
        """Test upload without deletion (delete=False)"""
        mock_s3 = _setup_s3_mock(mock_boto_client, [{'Contents': [{'Key': 'output/old_file.txt', 'Size': 100}]}])

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_text('content')

            options = Options(show_progress=False)
            with Mirror(
                options, output=Upload('s3://bucket/output', local=str(local_dir), interval=None, delete=False)
            ):
                pass

            # Should not delete anything
            mock_s3.delete_object.assert_not_called()

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_cache_directory_creation(self, mock_boto_client):
        """Test that cache directory is created when local=None"""
        _setup_s3_mock(mock_boto_client, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            options = Options(cache_root=tmpdir, show_progress=False)
            with Mirror(options, output=Upload('s3://bucket/output', interval=None)) as m:
                # Should create cache directory
                assert m.output.exists()
                assert m.output.is_dir()
                # Should be in cache
                assert str(m.output).startswith(tmpdir)


class TestBackgroundSync:
    """Test background sync functionality"""

    @patch(BOTO3_PATCH_TARGET)
    def test_background_sync_starts_and_stops(self, mock_boto_client):
        """Test that background sync thread starts and stops properly"""
        _setup_s3_mock(mock_boto_client, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()

            options = Options(show_progress=False)
            with Mirror(options, output=Upload('s3://bucket/output', local=str(local_dir), interval=1)) as mirror:
                # Single background thread should be running
                assert mirror._sync_thread is not None
                assert mirror._sync_thread.is_alive()

                # Give it time to potentially sync once
                time.sleep(1.5)

            # After exit, thread should be stopped
            assert not mirror._sync_thread.is_alive()

    @patch(BOTO3_PATCH_TARGET)
    def test_background_sync_periodic_uploads(self, mock_boto_client):
        """Test that background sync uploads periodically"""
        mock_s3 = _setup_s3_mock(mock_boto_client, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_text('content')

            options = Options(show_progress=False)
            with Mirror(options, output=Upload('s3://bucket/output', local=str(local_dir), interval=1)):
                # Wait for at least 2 sync cycles
                time.sleep(2.5)

            # Should have synced multiple times (background + final)
            # At least 2-3 times depending on timing
            assert mock_s3.upload_file.call_count >= 2


class TestMirrorContextManager:
    """Test Mirror context manager lifecycle"""

    @patch(BOTO3_PATCH_TARGET)
    def test_attribute_access(self, mock_boto_client):
        """Test accessing paths via attributes"""
        _setup_s3_mock(mock_boto_client, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            options = Options(cache_root=tmpdir, show_progress=False)
            with Mirror(
                options, dataset=Download('s3://bucket/data'), output=Upload('s3://bucket/output', interval=None)
            ) as m:
                # Should be able to access via attributes
                assert isinstance(m.dataset, Path)
                assert isinstance(m.output, Path)

                # Should raise AttributeError for non-existent
                with pytest.raises(AttributeError):
                    _ = m.nonexistent

    @patch(BOTO3_PATCH_TARGET)
    def test_exit_performs_final_sync(self, mock_boto_client):
        """Test that __exit__ performs final sync for all uploads"""
        mock_s3 = _setup_s3_mock(mock_boto_client, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_text('content')

            options = Options(show_progress=False)
            # No interval - should only sync on exit
            with Mirror(options, output=Upload('s3://bucket/output', local=str(local_dir), interval=None)):
                # During context, no uploads yet
                initial_count = mock_s3.upload_file.call_count

            # After exit, final sync should have happened
            assert mock_s3.upload_file.call_count > initial_count

    @patch(BOTO3_PATCH_TARGET)
    def test_multiple_uploads_and_downloads(self, mock_boto_client):
        """Test multiple uploads and downloads in same context"""
        mock_s3 = _setup_s3_mock(mock_boto_client)

        # Create separate paginators for each call
        def make_paginator(bucket, prefix):
            if 'data1' in prefix:
                return [{'Contents': [{'Key': 'data1/file.txt', 'Size': 100}]}]
            elif 'data2' in prefix:
                return [{'Contents': [{'Key': 'data2/file.txt', 'Size': 200}]}]
            else:
                return [{}]  # Empty result for uploads

        mock_paginator = Mock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = lambda Bucket, Prefix: make_paginator(Bucket, Prefix)

        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / 'out1'
            out1.mkdir()
            out2 = Path(tmpdir) / 'out2'
            out2.mkdir()

            options = Options(cache_root=tmpdir, show_progress=False)
            with Mirror(
                options,
                dataset1=Download('s3://bucket/data1'),
                dataset2=Download('s3://bucket/data2'),
                output1=Upload('s3://bucket/output1', local=str(out1), interval=None),
                output2=Upload('s3://bucket/output2', local=str(out2), interval=None),
            ) as m:
                # All paths should be accessible
                assert m.dataset1.exists()
                assert m.dataset2.exists()
                assert m.output1.exists()
                assert m.output2.exists()


class TestLocalPaths:
    """Test local path pass-through"""

    def test_download_local_path(self):
        """Test Download with local path (no S3 download)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_data = Path(tmpdir) / 'data'
            local_data.mkdir()
            (local_data / 'file.txt').write_text('content')

            options = Options(show_progress=False)
            with Mirror(options, dataset=Download(str(local_data))) as m:
                # Should return the local path directly
                assert m.dataset == local_data
                assert (m.dataset / 'file.txt').read_text() == 'content'

    def test_download_local_path_not_exists(self):
        """Test Download with non-existent local path raises error"""
        options = Options(show_progress=False)
        with pytest.raises(FileNotFoundError, match='Local download path does not exist'):
            with Mirror(options, dataset=Download('/nonexistent/path')):
                pass

    def test_upload_local_path(self):
        """Test Upload with local path (no S3 upload)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_output = Path(tmpdir) / 'output'

            options = Options(show_progress=False)
            with Mirror(options, output=Upload(str(local_output), interval=None)) as m:
                # Should return the local path directly
                assert m.output == local_output
                assert m.output.exists()
                assert m.output.is_dir()

    def test_mixed_s3_and_local(self):
        """Test mixing S3 and local paths in same Mirror"""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_data = Path(tmpdir) / 'local_data'
            local_data.mkdir()
            (local_data / 'file.txt').write_text('local content')

            local_output = Path(tmpdir) / 'local_output'

            options = Options(cache_root=tmpdir, show_progress=False)

            with patch(BOTO3_PATCH_TARGET) as mock_boto_client:
                mock_s3 = _setup_s3_mock(mock_boto_client, [{'Contents': [{'Key': 's3data/file.txt', 'Size': 100}]}])

                with Mirror(
                    options,
                    local_dataset=Download(str(local_data)),
                    s3_dataset=Download('s3://bucket/s3data'),
                    local_output=Upload(str(local_output), interval=None),
                    s3_output=Upload('s3://bucket/s3output', interval=None),
                ) as m:
                    # Local paths pass through
                    assert m.local_dataset == local_data
                    assert m.local_output == local_output

                    # S3 paths use cache
                    assert str(m.s3_dataset).startswith(tmpdir)
                    assert str(m.s3_output).startswith(tmpdir)

                # Should only download/upload S3 paths
                assert mock_s3.download_file.call_count >= 1
                # No uploads since local_output is local-only and s3_output is empty

    @patch(BOTO3_PATCH_TARGET)
    def test_no_s3_conflict_with_local(self, mock_boto_client):
        """Test that local paths don't conflict with S3 paths"""
        _setup_s3_mock(mock_boto_client, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / 'data'
            local_path.mkdir()

            options = Options(show_progress=False)
            # This should not raise ValueError (different path types)
            with Mirror(
                options, local_in=Download(str(local_path)), s3_out=Upload('s3://bucket/data', interval=None)
            ) as m:
                assert m.local_in == local_path


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_spec_type(self):
        """Test that invalid spec type raises TypeError"""
        with pytest.raises(TypeError, match='must be Download or Upload'):
            Mirror(dataset='invalid_spec')

    def test_download_upload_conflict_exact_match(self):
        """Test that same S3 path for download and upload raises ValueError"""
        with pytest.raises(ValueError, match='conflict'):
            Mirror(data_in=Download('s3://bucket/data'), data_out=Upload('s3://bucket/data', interval=None))

    def test_download_upload_conflict_subdirectory_download_parent(self):
        """Test that subdirectory conflict is detected (download parent, upload child)"""
        with pytest.raises(ValueError, match='conflict'):
            Mirror(data_in=Download('s3://bucket/data'), data_out=Upload('s3://bucket/data/subset', interval=None))

    def test_download_upload_conflict_subdirectory_upload_parent(self):
        """Test that subdirectory conflict is detected (upload parent, download child)"""
        with pytest.raises(ValueError, match='conflict'):
            Mirror(data_in=Download('s3://bucket/data/subset'), data_out=Upload('s3://bucket/data', interval=None))

    def test_download_upload_conflict_with_trailing_slash(self):
        """Test that conflict detection works with trailing slashes"""
        with pytest.raises(ValueError, match='conflict'):
            Mirror(data_in=Download('s3://bucket/data/'), data_out=Upload('s3://bucket/data/subset/', interval=None))

    def test_no_conflict_different_directories(self):
        """Test that different S3 directories don't conflict"""
        # This should not raise - different directories
        Mirror(data_in=Download('s3://bucket/data'), other_out=Upload('s3://bucket/data-new', interval=None))

    @patch(BOTO3_PATCH_TARGET)
    def test_download_failure_logged(self, mock_boto_client):
        """Test that download failures are logged but don't crash"""
        mock_s3 = _setup_s3_mock(mock_boto_client, [{'Contents': [{'Key': 'data/file1.txt', 'Size': 100}]}])

        # Make download fail
        mock_s3.download_file.side_effect = Exception('Download failed')

        with tempfile.TemporaryDirectory() as tmpdir:
            options = Options(cache_root=tmpdir, show_progress=False)
            # Should not crash, just log error
            with Mirror(options, dataset=Download('s3://bucket/data')):
                pass

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_failure_logged(self, mock_boto_client):
        """Test that upload failures are logged but don't crash"""
        mock_s3 = _setup_s3_mock(mock_boto_client, [])

        # Make upload fail
        mock_s3.upload_file.side_effect = Exception('Upload failed')

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'output'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_text('content')

            options = Options(show_progress=False)
            # Should not crash, just log error
            with Mirror(options, output=Upload('s3://bucket/output', local=str(local_dir), interval=None)):
                pass
