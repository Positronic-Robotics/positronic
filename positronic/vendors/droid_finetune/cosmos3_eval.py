"""Score Cosmos3-Nano weights on a held-out Cosmos3-DROID tree, through NVIDIA's RoboLab action server code.

Usage (in the train-cosmos3 image, with the parent of this package on PYTHONPATH)
  python -m droid_finetune.cosmos3_eval --checkpoint=/opt/cosmos3/dcp --heldout=<dir>/heldout \
      --output=<dir>/start.json
  python -m droid_finetune.cosmos3_eval --checkpoint=<job dir>/checkpoints/iter_000001000 --heldout=<dir>/heldout \
      --output=<dir>/iter_000001000.json

`--checkpoint` is a DCP checkpoint: the image's start weights, or one that a training run of
`action_policy_droid_nano` wrote. Every window samples with the same noise seed. The server's own defaults choose
the sampler and the prompt format, as when it serves the policy. It prints the summary of the report as one JSON line.
"""

import argparse
import os
import time
from pathlib import Path

from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

from . import heldout
from .report import build_report, write_report

EXPERIMENT = 'action_policy_droid_nano'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, required=True, help=f'the tree that holds {heldout.DATASET_DIR}/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows-per-episode', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vae-path', default=os.environ.get('WAN_VAE_PATH'), help='default: $WAN_VAE_PATH')
    parser.add_argument('--format-prompt-as-json', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--override', action='append', default=[], help='a Hydra override of the experiment')
    args = parser.parse_args()
    if args.vae_path is None:
        parser.error('--vae-path is required when WAN_VAE_PATH is not set')
    # The experiment's dataset reads DROID_ROOT when the config loads.
    os.environ.setdefault('DROID_ROOT', str(args.heldout / heldout.DATASET_DIR))

    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=str(args.checkpoint),
            allow_dcp_checkpoint=True,
            experiment=EXPERIMENT,
            experiment_overrides=[f'model.config.tokenizer.vae_path={args.vae_path}', *args.override],
            seed=args.seed,
            deterministic_seed=True,
            format_prompt_as_json=args.format_prompt_as_json,
        )
    )
    started = time.perf_counter()
    episodes = heldout.read_episodes(args.heldout)
    windows = heldout.plan_windows(episodes, args.windows_per_episode)
    scored, per_channel = heldout.score_windows(windows, lambda observation: service.infer(observation)['action'])
    report = build_report(
        scored,
        per_channel.tolist(),
        framework='cosmos3',
        checkpoint=str(args.checkpoint),
        heldout=str(args.heldout),
        seed=args.seed,
        windows_per_episode=args.windows_per_episode,
        episodes=len({w.episode.index for w in windows}),
        seconds=time.perf_counter() - started,
    )
    write_report(report, args.output)


if __name__ == '__main__':
    main()
