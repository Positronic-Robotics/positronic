import dataclasses
import importlib.util
import sys

import openpi.training.config as _config
from openpi.training.optimizer import CosineDecaySchedule
from wandb.sdk.wandb_run import Run


def _extract_openpi_root() -> str:
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != '--openpi-root':
        raise SystemExit('_launch.py expects `--openpi-root <path>` as its first argument')
    root = argv[1]
    sys.argv = [sys.argv[0], *argv[2:]]
    return root


def _load_openpi_train(openpi_root: str):
    spec = importlib.util.spec_from_file_location('openpi_train', f'{openpi_root}/scripts/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    openpi_root = _extract_openpi_root()
    openpi_train = _load_openpi_train(openpi_root)

    cfg = _config.cli()

    if isinstance(cfg.lr_schedule, CosineDecaySchedule):
        cfg = dataclasses.replace(
            cfg, lr_schedule=dataclasses.replace(cfg.lr_schedule, decay_steps=cfg.num_train_steps)
        )

    lr_fn = cfg.lr_schedule.create()

    # openpi calls `wandb.init()` inside its train main, which rebinds the
    # module-level `wandb.log`. Patch the `Run.log` chokepoint instead so the
    # injected learning_rate survives init regardless of patch timing.
    _orig_log = Run.log

    def _log(self, data, *args, **kwargs):
        step = kwargs.get('step', args[0] if args else None)
        if step is not None and isinstance(data, dict) and 'loss' in data:
            data = {**data, 'learning_rate': float(lr_fn(step))}
        return _orig_log(self, data, *args, **kwargs)

    Run.log = _log
    openpi_train.main(cfg)


if __name__ == '__main__':
    main()
