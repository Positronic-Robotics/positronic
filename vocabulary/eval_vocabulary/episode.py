"""Where an episode records its statics, and the keys that a scorer reads from them.

The dataset writer and a scorer both import these names, so they cannot disagree about them.
"""

# The file in an episode's directory that holds its statics as one JSON object.
STATIC_FILE = 'static.json'
# The success that the env reports when the trial ends.
SUCCESS = 'eval.success'
# The instruction that the policy got.
TASK = 'task'
