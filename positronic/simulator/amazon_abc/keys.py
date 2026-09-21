"""The keys of an ABC scene: the render settings the eval config owns.

``_reset_token`` reads them back beside the task name ``eval.keys.TASK`` carries. The server caches an env by
``(task, height, width)``.
"""

CAMERA_HEIGHT = 'abc.camera_height'
CAMERA_WIDTH = 'abc.camera_width'
