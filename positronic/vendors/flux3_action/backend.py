"""Serve FLUX 3 Action's DROID policy with Black Forest Labs' own server, for `server.py` on this machine.

It runs in BFL's own environment, so it imports nothing from positronic.

Usage
  python -P backend.py --checkpoint black-forest-labs/flux-3-action-droid --revision <commit> \
      --subfolder variants/gd [--port 9000]
"""

import argparse

import torch
from flux_action.serving import robolab


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve FLUX 3 Action's DROID policy with BFL's server.")
    parser.add_argument('--checkpoint', required=True, help='a Hugging Face repository id or a local directory')
    parser.add_argument('--revision', help='the repository commit')
    parser.add_argument('--subfolder', help='the policy package in the repository, e.g. variants/gd')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args(argv)

    # FOOTGUN: without this capture the warm-up dies on "Inplace update to inference tensor outside
    # InferenceMode" as it captures the DiT's CUDA graph, in `serve-robolab` too. The graph lives for the run.
    first_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(first_graph):
        torch.zeros(1, device='cuda')

    # The defaults of `serve-robolab`: DiT in bfloat16, compiled, one warm-up request before the bind.
    policy = robolab.load_serving_policy(
        args.checkpoint, revision=args.revision, subfolder=args.subfolder, device='cuda', dtype='bfloat16'
    )
    metadata = {
        'checkpoint': args.checkpoint,
        'revision': args.revision,
        'subfolder': args.subfolder,
        'serving_setup': policy.serving_setup,
    }
    # The bind comes after the warm-up, because an open port tells `server.py` that the model can serve.
    robolab.serve(robolab.RoboLabPolicy(policy), host=args.host, port=args.port, metadata=metadata)


if __name__ == '__main__':
    main()
