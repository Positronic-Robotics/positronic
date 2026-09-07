"""The keys of what a policy reports about itself, and where the harness records them."""

# What a policy reports about itself through its ``meta``; a remote policy nests the server's meta under
# ``SERVER``, and the harness records the result under ``POLICY_META``. ``TYPE`` names the policy at the top
# level and the vendor under ``SERVER``, so a reader composes a prefix with a field: f'{SERVER_META}.{TYPE}'.
TYPE = 'type'
CHECKPOINT_PATH = 'checkpoint_path'
EXPERIMENT_NAME = 'experiment_name'
CONFIG_NAME = 'config_name'
SERVER = 'server'

POLICY_META = 'inference.policy'
SERVER_META = f'{POLICY_META}.{SERVER}'
