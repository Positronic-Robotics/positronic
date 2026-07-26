import logging

import pos3

logger = logging.getLogger(__name__)


def list_checkpoints(checkpoints_dir: str, prefix: str = '') -> list[str]:
    """List available checkpoint subdirectories in a checkpoints directory.

    Expects numeric checkpoint directories, optionally prefixed (e.g. "checkpoint-123").
    Returns the matching directory names, sorted numerically.
    """
    checkpoint_nums: list[tuple[int, str]] = []
    children: list[str] = []
    for child in pos3.ls(checkpoints_dir, recursive=False):
        name = child.rstrip('/').split('/')[-1]
        children.append(name)
        if name.startswith(prefix):
            candidate = name[len(prefix) :]
            if candidate.isdigit():
                checkpoint_nums.append((int(candidate), name))

    if not checkpoint_nums:
        raise ValueError(f'No checkpoint found in {checkpoints_dir}. Available files: {children}')

    checkpoint_nums.sort(key=lambda pair: pair[0])
    return [name for _, name in checkpoint_nums]


def get_latest_checkpoint(checkpoints_dir: str, prefix: str = '') -> str:
    return list_checkpoints(checkpoints_dir, prefix=prefix)[-1]


def resolve_checkpoint(checkpoints_dir: str, configured: str | None, requested: str | None) -> str:
    """Resolve a checkpoint ID from an explicit request, a configured default, or latest available."""
    if requested:
        available = list_checkpoints(checkpoints_dir)
        if requested not in available:
            raise ValueError(f'Checkpoint not found: {requested}. Available: {available}')
        return requested

    if configured:
        checkpoint_id = str(configured).strip('/')
        available = list_checkpoints(checkpoints_dir)
        if checkpoint_id not in available:
            raise ValueError(f'Configured checkpoint not found: {checkpoint_id}. Available: {available}')
        logger.info(f'Using configured checkpoint: {checkpoint_id}')
        return checkpoint_id

    checkpoint_id = get_latest_checkpoint(checkpoints_dir)
    logger.info(f'Using latest checkpoint: {checkpoint_id}')
    return checkpoint_id
