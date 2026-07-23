"""Canonical raw observation keys of the positronic inference wire.

Server-side codecs consume these keys from the raw observation dict; every client — positronic's own
``RemotePolicy``, sim adapters, customer integrations — sends them. They are defined here, once, so a rename is a
one-site change the type checker propagates instead of a silent client/server desync.
"""

JOINTS = 'robot_state.q'
EE_POSE = 'robot_state.ee_pose'
GRIP = 'grip'
WRIST_IMAGE = 'image.wrist'
EXTERIOR_IMAGE = 'image.exterior'
TASK = 'task'
