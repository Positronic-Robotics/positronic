"""Fine-tune FLUX 3 Action from a policy package with its embodiment heads, and stop when the package lacks one.

`flux-action train` loads `policy.trunk_weights` with `strict_heads=False`: a head that the file lacks gets fresh
weights and no error. This entry point loads the same file with `strict_heads=True`.

Usage (in the train-flux3 image, with the parent of this package on PYTHONPATH)
  python -m droid_finetune.flux3_train trunk --package=/opt/flux3/droid --output=<run>/droid_trunk.safetensors
  torchrun --nproc_per_node=8 -m droid_finetune.flux3_train train --config=<run>/train.json \
      --step0-export=<output_dir>/step-0-export

`trunk` writes the weights of a package in the layout that `policy.trunk_weights` reads; `train.json` names that file.
`--step0-export` writes the warm-started weights, before the first update, as a bfloat16 policy package, once per
run, for `droid_finetune.flux3_eval`.
"""

import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path

import torch
from flux_action.config import PolicyConfig
from flux_action.models.wiring import load_action_checkpoint
from flux_action.policy import FluxActionPolicy, write_policy_export
from flux_action.training.trainer import TrainConfig, Trainer
from safetensors.torch import load_file, save_file

from .flux3_eval import PACKAGE_WEIGHTS, package_config

# A package holds the policy's state dict; the policy keeps its DiT under this prefix.
DIT_PREFIX = 'dit.'


def write_trunk(package: Path, output: Path) -> None:
    if package_config(package).get('quantization') is not None:
        raise ValueError(f'{package} holds quantized weights; train from a bfloat16 package')
    state = load_file(str(package / PACKAGE_WEIGHTS))
    foreign = [key for key in state if not key.startswith(DIT_PREFIX)]
    if foreign:
        raise ValueError(f'{package}: weights outside the DiT, e.g. {foreign[:3]}')
    partial = output.with_name(output.name + '.partial')
    save_file({key.removeprefix(DIT_PREFIX): value for key, value in state.items()}, str(partial))
    partial.replace(output)


def _export_bfloat16(policy: FluxActionPolicy, directory: Path) -> None:
    partial = directory.with_name(directory.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in policy.state_dict().items()}
    write_policy_export(partial, policy.config, state, weight_profile='model')
    partial.rename(directory)


def strict_warm_start(step0_export: Path | None):
    """The trainer's policy factory: the policy with every DiT weight, heads included, from `trunk_weights`."""

    def build(config: PolicyConfig) -> FluxActionPolicy:
        trunk = config.trunk_weights
        if trunk is None:
            raise ValueError('train.json names no policy.trunk_weights')
        policy = FluxActionPolicy(dataclasses.replace(config, trunk_weights=None))
        load_action_checkpoint(
            policy.dit, trunk, config.action_modality, strict_heads=True, head_seed=config.head_init_seed
        )
        policy.config.trunk_weights = trunk
        if step0_export is not None and int(os.environ.get('RANK', '0')) == 0 and not step0_export.exists():
            _export_bfloat16(policy, step0_export)
        return policy

    return build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    trunk = commands.add_parser('trunk', help='write the weights of a package as a trunk file')
    trunk.add_argument('--package', type=Path, required=True)
    trunk.add_argument('--output', type=Path, required=True)
    train = commands.add_parser('train', help='run `flux-action train` with the strict warm start')
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--override', action='append', default=[], metavar='KEY=VALUE')
    train.add_argument('--step0-export', type=Path, default=None)
    args = parser.parse_args()
    if args.command == 'trunk':
        write_trunk(args.package, args.output)
        return
    config = TrainConfig.from_file(args.config, args.override)
    result = Trainer(config, policy_factory=strict_warm_start(args.step0_export)).run()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
