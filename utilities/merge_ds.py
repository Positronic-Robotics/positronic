import glob
import os

import fire


def merge_ds(target_dir, *directories):
    """
    Merge first dataset into second dataset.

    Directory 1 has structure:
    - dir1/
        - f1.pt
        - f2.pt
        - ...
    Directory 2 has structure:
    - dir2/
        - f1.pt
        - f2.pt
        - ...
    Merged dataset has structure:
    - target_dir/
        - 001.pt
        - 002.pt
        - ...
        - N.pt    # Below is the samples from dataset 1
        - N+1.pt
        - ...
    """
    os.makedirs(target_dir, exist_ok=True)
    if os.listdir(target_dir):
        raise ValueError(f'Target directory {target_dir} is not empty')

    files_saved = 0

    for directory in directories:
        for f in sorted(glob.glob(os.path.join(directory, '*.pt'))):
            files_saved += 1  # enumerating from 1 to be consistent with dataset_dumper
            os.symlink(f, os.path.join(target_dir, f'{files_saved:03d}.pt'))


if __name__ == '__main__':
    fire.Fire(merge_ds)
