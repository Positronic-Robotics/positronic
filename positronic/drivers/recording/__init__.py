"""Recording agents and Harness timeline names."""

from eval_vocabulary.timeline import HARNESS_WALL as HARNESS_WALL
from eval_vocabulary.timeline import HARNESS_WORLD as HARNESS_WORLD

# The number of the step that a sample of the record goes with. The harness counts the steps of an episode from 0.
# TODO: move to eval_vocabulary.timeline beside HARNESS_WORLD, which owns the names of the harness's timelines. The
# move needs new versions of the vocabulary, positronic-evals and the platform client, and their pins.
HARNESS_STEP = 'harness.step'
