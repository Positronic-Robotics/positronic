"""The timelines that pimm stamps on each message, and that a recording keeps as time columns.

pimm writes these names and a labeler reads them back, so both import them from here.
"""

EMITTED_PREFIX = 'emitted.'
RECEIVED_PREFIX = 'received.'
EMITTED_WALL = f'{EMITTED_PREFIX}wall'
# The timeline of when each sample was made, on the world clock that a trial runs on.
EMITTED_WORLD = f'{EMITTED_PREFIX}world'
RECEIVED_WALL = f'{RECEIVED_PREFIX}wall'
RECEIVED_WORLD = f'{RECEIVED_PREFIX}world'
