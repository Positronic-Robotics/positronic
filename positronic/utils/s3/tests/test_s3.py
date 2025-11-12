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

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_no_sync_on_error(self, mock_boto_client):
        """Test that uploads with sync_on_error=False don't sync when context exits with error."""
        mock_s3 = _setup_s3_mock(mock_boto_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'output'
            output.mkdir()
            (output / 'data.txt').write_text('content')

            try:
                with s3.mirror(cache_root=tmpdir, show_progress=False):
                    s3.upload('s3://bucket/output', local=output, interval=None, sync_on_error=False)
                    raise RuntimeError('Test error')
            except RuntimeError:
                pass

        # Should not have synced because sync_on_error=False and context exited with error
        assert mock_s3.upload_file.call_count == 0

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_sync_on_error_true(self, mock_boto_client):
        """Test that uploads with sync_on_error=True do sync when context exits with error."""
        mock_s3 = _setup_s3_mock(mock_boto_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'output'
            output.mkdir()
            (output / 'data.txt').write_text('content')

            try:
                with s3.mirror(cache_root=tmpdir, show_progress=False):
                    s3.upload('s3://bucket/output', local=output, interval=None, sync_on_error=True)
                    raise RuntimeError('Test error')
            except RuntimeError:
                pass

        # Should have synced because sync_on_error=True
        assert mock_s3.upload_file.call_count >= 1


class TestDownloadSync:
    @patch(BOTO3_PATCH_TARGET)
    def test_download_delete_removes_orphaned_files(self, mock_boto_client):
        """Test that download with delete=True removes local files not in S3."""
        # S3 only has file1.txt
        paginate = [{'Contents': [{'Key': 'data/file1.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            # Create local files - file1.txt (exists in S3) and file2.txt (orphan)
            (local_dir / 'file1.txt').write_bytes(b'12345')
            orphan_file = local_dir / 'file2.txt'
            orphan_file.write_text('orphan')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, delete=True)

            # file1.txt should exist (no need to download, same size)
            assert (local_dir / 'file1.txt').exists()
            # file2.txt should be deleted
            assert not orphan_file.exists()

    @patch(BOTO3_PATCH_TARGET)
    def test_download_no_delete_preserves_orphaned_files(self, mock_boto_client):
        """Test that download with delete=False preserves local files not in S3."""
        # S3 only has file1.txt
        paginate = [{'Contents': [{'Key': 'data/file1.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            # Create local files
            (local_dir / 'file1.txt').write_bytes(b'12345')
            orphan_file = local_dir / 'file2.txt'
            orphan_file.write_text('orphan')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, delete=False)

            # Both files should exist
            assert (local_dir / 'file1.txt').exists()
            assert orphan_file.exists()

    @patch(BOTO3_PATCH_TARGET)
    def test_download_syncs_directories(self, mock_boto_client):
        """Test that download syncs directory structure including empty dirs."""
        # S3 has a directory marker
        paginate = [
            {
                'Contents': [
                    {'Key': 'data/subdir/', 'Size': 0},  # Directory marker
                    {'Key': 'data/subdir/file.txt', 'Size': 5},
                ]
            }
        ]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, delete=True)

            # Directory should be created
            assert (local_dir / 'subdir').is_dir()
            # File download should have been attempted
            assert mock_s3.download_file.call_count >= 1

    @patch(BOTO3_PATCH_TARGET)
    def test_download_delete_parameter_conflict(self, mock_boto_client):
        """Test that registering same download with different delete param raises error."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', delete=True)
                with pytest.raises(ValueError, match='already registered with different parameters'):
                    s3.download('s3://bucket/data', delete=False)

    @patch(BOTO3_PATCH_TARGET)
    def test_download_delete_empty_s3_removes_all_local(self, mock_boto_client):
        """Test that download with delete=True removes all local files, dirs, and root when S3 is empty."""
        # S3 is completely empty
        paginate = [{'Contents': []}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            # Create nested directory structure with files
            (local_dir / 'file.txt').write_text('content')
            subdir = local_dir / 'subdir'
            subdir.mkdir()
            (subdir / 'nested.txt').write_text('nested content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, delete=True)

            # When S3 is completely empty, the local directory itself should be removed
            assert not local_dir.exists()

    @patch(BOTO3_PATCH_TARGET)
    def test_download_no_delete_empty_s3_preserves_local(self, mock_boto_client):
        """Test that download with delete=False preserves local dir when S3 is empty."""
        # S3 is completely empty
        paginate = [{'Contents': []}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file.txt').write_text('content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, delete=False)

            # With delete=False, local directory and its contents should be preserved
            assert local_dir.exists()
            assert (local_dir / 'file.txt').exists()


class TestSync:
    @patch(BOTO3_PATCH_TARGET)
    def test_sync_requires_active_mirror(self, mock_boto_client):
        """Test that sync requires an active mirror context."""
        with pytest.raises(RuntimeError, match='No active mirror'):
            s3.sync('s3://bucket/data')

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_basic_functionality(self, mock_boto_client):
        """Test that sync performs download then upload and allows same remote path."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            # Add a new file that doesn't exist in S3 to ensure upload happens
            (local_dir / 'new_file.txt').write_text('new content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                result_path = s3.sync('s3://bucket/data', local=local_dir, interval=None, delete_local=False)

            assert result_path == local_dir.resolve()
            # Should have downloaded
            assert mock_s3.download_file.call_count >= 1
            # Should have uploaded (at least the new file)
            assert mock_s3.upload_file.call_count >= 1

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_local_passthrough(self, mock_boto_client):
        """Test that sync with local path just returns the path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / 'data'
            local_path.mkdir()

            with s3.mirror(show_progress=False):
                resolved = s3.sync(str(local_path))

            assert resolved == local_path.resolve()

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_delete_flags(self, mock_boto_client):
        """Test that delete_local and delete_remote flags work correctly."""
        # Test delete_local: S3 only has file1.txt
        paginate_local = [{'Contents': [{'Key': 'data/file1.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate_local)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_bytes(b'12345')
            orphan_local = local_dir / 'file2.txt'
            orphan_local.write_text('orphan')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                # delete_local=True should remove orphaned local files
                s3.sync('s3://bucket/data', local=local_dir, delete_local=True, delete_remote=False, interval=None)

            assert (local_dir / 'file1.txt').exists()
            assert not orphan_local.exists()

        # Test delete_local=False: preserve orphaned files
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_bytes(b'12345')
            orphan_local = local_dir / 'file2.txt'
            orphan_local.write_text('orphan')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.sync('s3://bucket/data', local=local_dir, delete_local=False, delete_remote=False, interval=None)

            assert (local_dir / 'file1.txt').exists()
            assert orphan_local.exists()

        # Test delete_remote: S3 has file1.txt and file2.txt, local only has file1.txt
        paginate_remote = [{'Contents': [{'Key': 'data/file1.txt', 'Size': 5}, {'Key': 'data/file2.txt', 'Size': 5}]}]
        mock_s3_remote = _setup_s3_mock(mock_boto_client, paginate_remote)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_bytes(b'12345')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                # delete_remote=True should remove orphaned S3 files
                s3.sync('s3://bucket/data', local=local_dir, delete_local=False, delete_remote=True, interval=None)

            assert mock_s3_remote.delete_object.call_count >= 1

        # Test delete_remote=False: preserve orphaned S3 files
        mock_s3_no_delete = _setup_s3_mock(mock_boto_client, paginate_remote)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file1.txt').write_bytes(b'12345')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.sync('s3://bucket/data', local=local_dir, delete_local=False, delete_remote=False, interval=None)

            assert mock_s3_no_delete.delete_object.call_count == 0

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_background_sync(self, mock_boto_client):
        """Test that sync with interval enables background syncing."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file.txt').write_text('content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.sync('s3://bucket/data', local=local_dir, interval=1)
                time.sleep(2.5)

            # Should have synced multiple times in background
            assert mock_s3.upload_file.call_count >= 2

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_on_error_flag(self, mock_boto_client):
        """Test that sync_on_error flag controls syncing on context exit with error."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        mock_s3_no_sync = _setup_s3_mock(mock_boto_client, paginate)

        # Test sync_on_error=False: should not sync on error exit
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file.txt').write_text('content')

            try:
                with s3.mirror(cache_root=tmpdir, show_progress=False):
                    s3.sync('s3://bucket/data', local=local_dir, interval=None, sync_on_error=False)
                    raise RuntimeError('Test error')
            except RuntimeError:
                pass

            assert mock_s3_no_sync.download_file.call_count >= 1

        # Test sync_on_error=True: should sync on error exit
        mock_s3_with_sync = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file.txt').write_text('content')

            try:
                with s3.mirror(cache_root=tmpdir, show_progress=False):
                    s3.sync('s3://bucket/data', local=local_dir, interval=None, sync_on_error=True)
                    raise RuntimeError('Test error')
            except RuntimeError:
                pass

            # Should have synced because sync_on_error=True
            assert mock_s3_with_sync.upload_file.call_count >= 1

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_empty_s3_deletes_local_directories(self, mock_boto_client):
        """Test that sync with empty S3 deletes local directory completely."""
        # S3 is completely empty
        paginate = [{'Contents': []}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            # Create nested directory structure
            (local_dir / 'file.txt').write_text('content')
            subdir = local_dir / 'subdir'
            subdir.mkdir()
            (subdir / 'nested.txt').write_text('nested')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                # sync with empty S3 should delete everything local including root
                s3.sync('s3://bucket/data', local=local_dir, interval=None, delete_local=True, delete_remote=False)

            # When S3 is completely empty, the local directory itself should be removed
            assert not local_dir.exists()

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_conflicts(self, mock_boto_client):
        """Test that sync conflicts with existing registrations and second sync call."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        # Test conflict with existing download
        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data/subdir')
                with pytest.raises(ValueError, match='Conflict'):
                    s3.sync('s3://bucket/data', interval=None)

        # Test conflict with existing upload
        _setup_s3_mock(mock_boto_client)
        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.upload('s3://bucket/data/subdir')
                with pytest.raises(ValueError, match='Conflict'):
                    s3.sync('s3://bucket/data', interval=None)

        # Test second sync call conflicts (upload already registered)
        _setup_s3_mock(mock_boto_client, paginate)
        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                path1 = s3.sync('s3://bucket/data', interval=None)
                assert path1 is not None
                # Second sync call tries to download, which conflicts with existing upload
                with pytest.raises(ValueError, match='Conflict'):
                    s3.sync('s3://bucket/data', interval=None)


class TestLs:
    def test_ls_local_non_recursive(self):
        """Test non-recursive listing excludes nested items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / 'file.txt').write_text('x')
            (base / 'dir').mkdir()
            (base / 'dir' / 'nested.txt').write_text('x')

            with s3.mirror(show_progress=False):
                items = s3._require_active_mirror().ls(str(base), recursive=False)

            assert str(base / 'dir' / 'nested.txt') not in items
            assert str(base / 'file.txt') in items

    def test_ls_local_recursive(self):
        """Test recursive listing includes nested items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / 'dir').mkdir()
            (base / 'dir' / 'nested.txt').write_text('x')

            with s3.mirror(show_progress=False):
                items = s3._require_active_mirror().ls(str(base), recursive=True)

            assert str(base / 'dir' / 'nested.txt') in items

    @patch(BOTO3_PATCH_TARGET)
    def test_ls_s3_non_recursive(self, mock_boto_client):
        """Test non-recursive S3 listing excludes nested items."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}, {'Key': 'data/sub/nested.txt', 'Size': 10}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with s3.mirror(show_progress=False):
            items = s3._require_active_mirror().ls('s3://bucket/data', recursive=False)

        assert 's3://bucket/data/sub/nested.txt' not in items
        assert 's3://bucket/data/file.txt' in items

    @patch(BOTO3_PATCH_TARGET)
    def test_ls_s3_recursive(self, mock_boto_client):
        """Test recursive S3 listing includes nested items."""
        paginate = [{'Contents': [{'Key': 'data/sub/nested.txt', 'Size': 10}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with s3.mirror(show_progress=False):
            items = s3._require_active_mirror().ls('s3://bucket/data', recursive=True)

        assert 's3://bucket/data/sub/nested.txt' in items

    @patch(BOTO3_PATCH_TARGET)
    def test_ls_s3_no_spurious_prefix_match(self, mock_boto_client):
        """Test that listing s3://bucket/data doesn't match s3://bucket/data-other."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}, {'Key': 'data-other/file.txt', 'Size': 10}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with s3.mirror(show_progress=False):
            items = s3._require_active_mirror().ls('s3://bucket/data', recursive=False)

        assert 's3://bucket/data/file.txt' in items
        assert 's3://bucket/data-other/file.txt' not in items


class TestExclude:
    @patch(BOTO3_PATCH_TARGET)
    def test_download_exclude_simple_pattern(self, mock_boto_client):
        """Test that exclude filters out files matching simple patterns."""
        # S3 has file.txt and file.log
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}, {'Key': 'data/file.log', 'Size': 10}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, exclude=['*.log'])

            # Should only download file.txt, not file.log
            assert mock_s3.download_file.call_count == 1
            call_args = mock_s3.download_file.call_args_list[0][0]
            assert 'file.txt' in call_args[1]  # S3 key
            assert 'file.log' not in str(call_args)

    @patch(BOTO3_PATCH_TARGET)
    def test_download_exclude_recursive_pattern(self, mock_boto_client):
        """Test that exclude with ** filters recursively."""
        # S3 has files in nested directories
        paginate = [
            {
                'Contents': [
                    {'Key': 'data/file.txt', 'Size': 5},
                    {'Key': 'data/logs/error.log', 'Size': 10},
                    {'Key': 'data/logs/debug.log', 'Size': 10},
                    {'Key': 'data/sub/logs/info.log', 'Size': 10},
                ]
            }
        ]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, exclude=['**/*.log'])

            # Should only download file.txt, not any .log files
            assert mock_s3.download_file.call_count == 1
            call_args = mock_s3.download_file.call_args_list[0][0]
            assert 'file.txt' in call_args[1]

    @patch(BOTO3_PATCH_TARGET)
    def test_download_exclude_directory(self, mock_boto_client):
        """Test that excluding a directory excludes all its contents."""
        # S3 has files in multiple directories
        paginate = [
            {
                'Contents': [
                    {'Key': 'data/file.txt', 'Size': 5},
                    {'Key': 'data/logs/', 'Size': 0},  # Directory marker
                    {'Key': 'data/logs/error.log', 'Size': 10},
                    {'Key': 'data/logs/debug.log', 'Size': 10},
                ]
            }
        ]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, exclude=['logs'])

            # Should only download file.txt, not logs directory or its contents
            assert mock_s3.download_file.call_count == 1
            call_args = mock_s3.download_file.call_args_list[0][0]
            assert 'file.txt' in call_args[1]

    @patch(BOTO3_PATCH_TARGET)
    def test_upload_exclude_pattern(self, mock_boto_client):
        """Test that exclude filters out files during upload."""
        mock_s3 = _setup_s3_mock(mock_boto_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file.txt').write_text('content')
            (local_dir / 'file.log').write_text('log content')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.upload('s3://bucket/data', local=local_dir, interval=None, exclude=['*.log'])

            # Should only upload file.txt, not file.log
            assert mock_s3.upload_file.call_count == 1
            call_args = mock_s3.upload_file.call_args_list[0][0]
            assert 'file.txt' in str(call_args[0])  # Local file path
            assert 'file.log' not in str(call_args)

    @patch(BOTO3_PATCH_TARGET)
    def test_sync_exclude_pattern(self, mock_boto_client):
        """Test that exclude filters files during sync in both directions."""
        # S3 has file.txt and remote.log
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}, {'Key': 'data/remote.log', 'Size': 10}]}]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'
            local_dir.mkdir()
            (local_dir / 'file.txt').write_bytes(b'12345')  # Same size as S3
            (local_dir / 'local.log').write_text('local log')

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.sync('s3://bucket/data', local=local_dir, interval=None, exclude=['*.log'])

            # Should not download remote.log or upload local.log
            # Only file.txt should be considered (and it's already synced)
            assert mock_s3.download_file.call_count == 0  # file.txt already exists with same size
            assert mock_s3.upload_file.call_count == 0  # file.txt already synced, *.log excluded

    @patch(BOTO3_PATCH_TARGET)
    def test_download_exclude_parameter_conflict(self, mock_boto_client):
        """Test that registering download with different exclude param raises error."""
        paginate = [{'Contents': [{'Key': 'data/file.txt', 'Size': 5}]}]
        _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', exclude=['*.log'])
                with pytest.raises(ValueError, match='already registered with different parameters'):
                    s3.download('s3://bucket/data', exclude=['*.txt'])

    @patch(BOTO3_PATCH_TARGET)
    def test_exclude_multiple_patterns(self, mock_boto_client):
        """Test that multiple exclude patterns work together."""
        paginate = [
            {
                'Contents': [
                    {'Key': 'data/file.txt', 'Size': 5},
                    {'Key': 'data/file.log', 'Size': 10},
                    {'Key': 'data/file.tmp', 'Size': 10},
                ]
            }
        ]
        mock_s3 = _setup_s3_mock(mock_boto_client, paginate)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / 'data'

            with s3.mirror(cache_root=tmpdir, show_progress=False):
                s3.download('s3://bucket/data', local=local_dir, exclude=['*.log', '*.tmp'])

            # Should only download file.txt
            assert mock_s3.download_file.call_count == 1
            call_args = mock_s3.download_file.call_args_list[0][0]
            assert 'file.txt' in call_args[1]
