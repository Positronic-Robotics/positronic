"""Implementation of multiprocessing channels."""

from multiprocessing import Queue
from ironic2.channel import Channel, Message, NoValue, LastValueChannel


class QueueChannel(Channel):

    def __init__(self, max_size: int = 1):
        super().__init__()
        self.queue = Queue(max_size)

    def write(self, message: Message):
        if self.queue.full():
            self.queue.get()
        self.queue.put(message)

    def read(self):
        if self.queue.empty():
            return NoValue
        return self.queue.get()


def last_value_queue_channel(max_size: int = 1):
    return LastValueChannel(QueueChannel(max_size))
