"""Implementation of multiprocessing channels."""

from multiprocessing import Queue
from ironic2.channel import Channel, Message, NoValue, LastValueChannel


class QueueChannel(Channel):
    """Channel implementation using multiprocessing Queue."""

    def __init__(self, max_size: int = 1):
        """Initialize queue channel with maximum size."""
        super().__init__()
        self.queue = Queue(max_size)

    def write(self, message: Message):
        """Write message to queue. Drops oldest message if queue is full."""
        if self.queue.full():
            self.queue.get()
        self.queue.put(message)

    def read(self):
        """Read message from queue. Returns NoValue if queue is empty."""
        if self.queue.empty():
            return NoValue
        return self.queue.get()


def last_value_queue_channel(max_size: int = 1):
    """Create a LastValueChannel wrapping a QueueChannel."""
    return LastValueChannel(QueueChannel(max_size))
