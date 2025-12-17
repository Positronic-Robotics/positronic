import pos3


def get_latest_checkpoint(checkpoints_dir: str, prefix: str = '') -> str:
    checkpoint_nums = []
    children = []
    for child in pos3.ls(checkpoints_dir, recursive=False):
        name = child.rstrip('/').split('/')[-1]
        children.append(name)
        if name.startswith(prefix):
            candidate = name[len(prefix) :]
            if candidate.isdigit():
                checkpoint_nums.append(int(candidate))
    if checkpoint_nums:
        return prefix + str(max(checkpoint_nums))
    else:
        raise ValueError(f'No checkpoint found in {checkpoints_dir}. Available files: {children}')
