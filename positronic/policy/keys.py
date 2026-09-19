"""The keys of what a policy reports about itself, and where the harness records them."""

# What a policy reports about itself through its ``meta``; a remote policy nests the server's meta under
# ``SERVER``, and the harness records the result under ``POLICY_META``. ``TYPE`` names the policy at the top
# level and the vendor under ``SERVER``, so a reader composes a prefix with a field: f'{SERVER_META}.{TYPE}'.
# The block under ``SERVER`` is the handshake metadata as the server sent it. The client reads its declared
# stack and ``compress_images`` from it and records the rest without a check. A ``prompt`` in it is the
# deployment's own declaration, fixed for that deployment. The instruction an episode sent is ``keys.TASK``.
TYPE = 'type'
CHECKPOINT_PATH = 'checkpoint_path'
EXPERIMENT_NAME = 'experiment_name'
CONFIG_NAME = 'config_name'
SERVER = 'server'

POLICY_META = 'inference.policy'
SERVER_META = f'{POLICY_META}.{SERVER}'
