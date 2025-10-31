# S3 Mirror

Context manager for syncing directories and files with S3. Supports both S3 URLs and local paths - only remote paths are synced.

## Quick Start

```python
from positronic.utils.s3 import Mirror, Download, Upload

# Mirror downloads on enter, uploads on exit, and optionally syncs in background
with Mirror(
    dataset=Download('s3://bucket/data'),
    checkpoints=Upload('s3://bucket/outputs', local='./outputs')
) as m:
    print(m.dataset)      # Path to local copy: ~/.cache/positronic/s3/bucket/data
    print(m.checkpoints)  # Path to local dir: ./outputs

    # dataset is downloaded and ready to use
    # checkpoints directory will sync to S3 in background and on exit
    train(m.dataset, m.checkpoints)
```

## API

### Mirror

```python
Mirror(options=None, **paths)
```

Context manager that downloads on enter, starts background sync if configured, and performs final upload on exit.

**Arguments:**
- `options`: Optional `Options` instance
- `**paths`: Named `Download` or `Upload` specs

### Download

```python
Download(remote, local=None)
```

Download spec for S3 or local paths. **Arguments:** `remote` (S3 URL like `s3://bucket/key` or local path), `local` (optional local path, defaults to auto-generated cache path).

```python
Download('s3://bucket/data')                    # Auto cache: ~/.cache/positronic/s3/bucket/data
Download('s3://bucket/data', local='./data')    # Custom location
Download('/path/to/local/data')                 # Local path (no download)
```

### Upload

```python
Upload(remote, local=None, interval=300, delete=True)
```

Upload spec for S3 or local paths. **Arguments:** `remote` (S3 URL or local path), `local` (optional local path, defaults to auto-generated cache path), `interval` (background sync interval in seconds, `None` to disable), `delete` (delete S3 files not present locally).

```python
Upload('s3://bucket/output', interval=300)      # Background sync every 5 min
Upload('s3://bucket/output', interval=None)     # Only sync on exit
Upload('s3://bucket/output', delete=False)      # Don't delete remote files
Upload('/path/to/local/output')                 # Local path (no upload)
```

### Options

```python
Options(cache_root='~/.cache/positronic/s3/', show_progress=True, max_workers=10)
```

Configuration for cache location, progress bars, and parallel workers.

## Examples

### Training with S3 Dataset

```python
with Mirror(
    dataset=Download('s3://my-bucket/training-data'),
    checkpoints=Upload('s3://my-bucket/checkpoints', interval=300)
) as m:
    train_model(data_path=m.dataset, checkpoint_path=m.checkpoints)
```

### Custom Cache Location

```python
from positronic.utils.s3 import Options

opts = Options(cache_root='/mnt/fast-ssd/cache')
with Mirror(options=opts, data=Download('s3://bucket/data')) as m:
    process(m.data)  # Downloads to /mnt/fast-ssd/cache/bucket/data
```

## Notes

- **Sync logic**: Compares file sizes to determine if sync is needed
- **Parallel transfers**: Configurable worker count via `Options.max_workers`
- **Path conflicts**: S3 paths cannot overlap between downloads and uploads
- **Background sync**: Single thread handles all uploads according to their intervals
