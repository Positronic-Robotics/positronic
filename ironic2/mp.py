"""Implementation of multiprocessing channels."""

import multiprocessing as mp
import signal
import sys
from multiprocessing import Queue
from typing import Callable, List

from ironic2.channel import Channel, LastValueChannel, Message, NoValue


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


def _background_process_wrapper(control_loop: Callable, stopped: mp.Event, *channels):
    """Module-level wrapper function that can be pickled for multiprocessing."""
    try:
        control_loop(stopped, *channels)
    except KeyboardInterrupt:
        # Silently handle KeyboardInterrupt in background processes
        pass
    except Exception as e:
        print(f"Error in control loop: {e}")
        stopped.set()


class MPWorld:
    """
    A multiprocessing "World" that manages background processes and provides graceful shutdown.

    This class allows you to run multiple background loops in separate processes while
    coordinating their execution and ensuring clean shutdown when the main process exits.
    """

    stopped: mp.Event
    background_loops: List[mp.Process]

    def __init__(self):
        self.stopped = mp.Event()
        self.background_loops = []

    def add_background_loop(self, control_loop: Callable, *channels: List[Channel]):
        """
        Add a background loop to be executed in a separate process.

        Args:
            control_loop: A callable that takes (stopped_event, *channels) as arguments
            *channels: Variable number of Channel objects to pass to the control loop

        Returns:
            mp.Process: The created process object (not yet started)
        """
        process = mp.Process(
            target=_background_process_wrapper,
            args=(control_loop, self.stopped, *channels),
            daemon=True
        )
        self.background_loops.append(process)
        return process

    def run(self, main_loop, *channels: List[Channel]):
        """
        Start all background processes and run the main loop.

        This method sets up signal handlers for graceful shutdown, starts all
        background processes, and then runs the main loop. It ensures proper
        cleanup of all processes when the main loop exits or is interrupted.

        Args:
            main_loop: A callable that takes (stopped_event, *channels) as arguments
            *channels: Variable number of Channel objects to pass to the main loop
        """
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print("\nProgram interrupted by user, stopping...")
            self.stopped.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        for process in self.background_loops:
            process.start()

        try:
            main_loop(self.stopped, *channels)
        except KeyboardInterrupt:
            print("\nProgram interrupted by user, stopping...")
        finally:
            self.stopped.set()
            for process in self.background_loops:
                process.join(timeout=3)
