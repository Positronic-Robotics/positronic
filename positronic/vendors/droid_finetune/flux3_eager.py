"""`flux-action` with the policy's inference compile off.

Usage (in the train-flux3 image): python -m droid_finetune.flux3_eager evaluate <the flux-action evaluate arguments>

`flux-action evaluate` compiles the DROID backend with CUDA graphs, and the capture fails on the inference tensors
of an evaluation. Every evaluation runs eager, so the start weights and each checkpoint see the same kernels.
"""

import functools
import runpy

from flux_action.policy import FluxActionPolicy

FluxActionPolicy.prepare_inference = functools.partialmethod(FluxActionPolicy.prepare_inference, compile=False)

if __name__ == '__main__':
    runpy.run_module('flux_action.cli', run_name='__main__', alter_sys=True)
