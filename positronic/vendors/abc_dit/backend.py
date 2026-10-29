"""Serve an ABC-DiT checkpoint with ABC's own policy server, for `server.py` on this machine.

It runs in ABC's interpreter, so it imports nothing from positronic. ABC's `deploy/serve_policy.py` binds every
interface and takes no host, and the server it starts asks for no token, so this script binds it to loopback.

Usage
  python -P backend.py --policy.checkpoint-path=<path> --policy-type=dit [--policy.no-fast-inference] --port=<port>
"""

import logging
import os

import tyro
from deploy.serve_policy import create_policy
from deploy.serve_policy_config import Args
from deploy.websocket_server import WebsocketPolicyServer

HOST = '127.0.0.1'


def main(args: Args) -> None:
    # The log level of `deploy/serve_policy.py`.
    logging.basicConfig(level=logging.INFO if os.environ.get('DEPLOY_VERBOSE') else logging.WARNING, force=True)
    WebsocketPolicyServer(policy=create_policy(args), host=HOST, port=args.port).serve_forever()


if __name__ == '__main__':
    main(tyro.cli(Args))
