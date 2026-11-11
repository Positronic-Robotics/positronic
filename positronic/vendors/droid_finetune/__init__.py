"""Held-out evaluation and a strict warm start for DROID policy fine-tunes.

The package runs inside a training image, where positronic is not installed. Each image carries it at
/opt/train/python, which is on `PYTHONPATH`. docker/README.md § Held-out evaluation gives the commands.
"""
