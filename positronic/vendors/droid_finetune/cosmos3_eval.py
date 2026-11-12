"""Score Cosmos3-Nano weights on a held-out Cosmos3-DROID tree, through NVIDIA's RoboLab action server code.

Usage (in the train-cosmos3 image, with the parent of this package on PYTHONPATH)
  python -m droid_finetune.cosmos3_eval --checkpoint=/opt/cosmos3/dcp --heldout=<dir>/heldout \
      --output=<dir>/start.json
  python -m droid_finetune.cosmos3_eval --checkpoint=<job dir>/checkpoints/iter_000001000 --heldout=<dir>/heldout \
      --output=<dir>/iter_000001000.json

`--checkpoint` is a DCP checkpoint: the image's start weights, or one that a training run of
`action_policy_droid_nano` wrote. Every window samples with the same noise seed and the same kernels, so two runs
on one checkpoint give the same report. The prompt is JSON, as `action_policy_droid_nano` trains, and the server's
defaults choose the sampler. It prints the summary of the report as one JSON line.
"""

import argparse
import os
import time
from pathlib import Path

from . import heldout
from .report import build_report, write_report

EXPERIMENT = 'action_policy_droid_nano'
# The deterministic mode of cosmos_framework/scripts/train.py. The experiment turns on cudnn autotuning, and the
# server compiles the model; both choose kernels by their measured speed, so two runs can use different kernels.
DETERMINISTIC_ENV = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'FLASH_ATTENTION_DETERMINISTIC': '1'}
DETERMINISTIC_OVERRIDES = ('trainer.cudnn.deterministic=True', 'trainer.cudnn.benchmark=False')
# The guardrails download gated weights and filter no action. The image's start checkpoint holds no EMA weights,
# so every checkpoint is scored on its regular weights.
SETUP_UPDATE = {'guardrails': False, 'use_ema_weights': False, 'use_torch_compile': False}


def make_deterministic(environ, torch) -> None:
    """Call before the first CUDA call: cuBLAS reads its workspace setting once."""
    environ.update(DETERMINISTIC_ENV)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment_overrides(vae_path: str, extra: list[str]) -> list[str]:
    return [
        f'model.config.tokenizer.vae_path={vae_path}',
        'model.config.ema.enabled=False',
        *DETERMINISTIC_OVERRIDES,
        *extra,
    ]


def policy_service(server_args):
    """NVIDIA's RoboLab policy server with `SETUP_UPDATE` applied. Only the training image holds its code."""
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService  # noqa: PLC0415

    class OfflinePolicyService(RobolabPolicyService):
        def _build_setup_args(self, args):
            return super()._build_setup_args(args).model_copy(update=SETUP_UPDATE)

    return OfflinePolicyService(server_args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, required=True, help=f'the tree that holds {heldout.DATASET_DIR}/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows-per-episode', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vae-path', default=os.environ.get('WAN_VAE_PATH'), help='default: $WAN_VAE_PATH')
    # The server finds no dataset config in a DCP checkpoint of this experiment and builds plain-text prompts unless
    # it is told; the experiment's dataset sets format_prompt_as_json=True.
    parser.add_argument('--format-prompt-as-json', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--override', action='append', default=[], help='a Hydra override of the experiment')
    args = parser.parse_args(argv)
    if args.vae_path is None:
        parser.error('--vae-path is required when WAN_VAE_PATH is not set')
    return args


def main() -> None:
    args = parse_args()
    # The experiment's dataset reads DROID_ROOT when the config loads; it names the tree that is scored.
    os.environ['DROID_ROOT'] = str(args.heldout / heldout.DATASET_DIR)
    # Only the training image installs torch and cosmos_framework; the tests run without them.
    import torch  # noqa: PLC0415
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs  # noqa: PLC0415

    make_deterministic(os.environ, torch)
    service = policy_service(
        RobolabServerArgs(
            checkpoint_path=str(args.checkpoint),
            allow_dcp_checkpoint=True,
            experiment=EXPERIMENT,
            experiment_overrides=experiment_overrides(args.vae_path, args.override),
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
        seconds=time.perf_counter() - started,
    )
    write_report(report, args.output)


if __name__ == '__main__':
    main()
