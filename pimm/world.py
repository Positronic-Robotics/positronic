"""Implementation of multiprocessing channels."""

import heapq
import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import sys
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from enum import IntEnum
from multiprocessing import resource_tracker
from multiprocessing.managers import ValueProxy
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full
from typing import TypeVar

from .core import (
    Clock,
    ControlLoop,
    ControlSystem,
    ControlSystemEmitter,
    ControlSystemReceiver,
    FakeEmitter,
    FakeReceiver,
    Message,
    SignalEmitter,
    SignalReceiver,
    Sleep,
)
from .shared_memory import SMCompliant

T = TypeVar('T')


class TransportMode(IntEnum):
    UNDECIDED = 0
    QUEUE = 1
    SHARED_MEMORY = 2


class QueueEmitter(SignalEmitter[T]):
    def __init__(self, queue: mp.Queue, clock: Clock):
        self._queue = queue
        self._clock = clock

    def emit(self, data: T, ts: int = -1):
        ts = ts if ts >= 0 else self._clock.now_ns()
        try:
            self._queue.put_nowait(Message(data, ts))
        except Full:
            # Queue is full, try to remove old message and try again
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(Message(data, ts))
            except (Empty, Full):
                pass


class MultiprocessEmitter(SignalEmitter[T]):
    """Signal emitter that transparently bridges processes.

    The emitter owns both the queue transport and (when selected) a
    shared-memory buffer. It defers the transport choice until the first payload
    unless ``forced_mode`` pins the decision.

    Broadcast emitting is supported by allowing queues, up_values and sm_queues be lists.
    """

    def __init__(
        self,
        clock: Clock,
        queues: list[Queue],
        mode_value: mp.Value,
        lock: mp.Lock,
        ts_value: mp.Value,
        up_values: list[ValueProxy[bool]],
        sm_queues: list[Queue],
        *,
        forced_mode: TransportMode | None = None,
    ):
        self._clock = clock
        self._queues = queues
        self._mode_value = mode_value
        self._forced_mode = forced_mode
        self._mode = forced_mode or TransportMode.UNDECIDED

        # Shared memory state
        self._data_type: type[SMCompliant] | None = None
        self._lock = lock
        self._ts_value = ts_value
        self._up_values = up_values
        self._sm_queues = sm_queues
        self._sm: multiprocessing.shared_memory.SharedMemory | None = None
        self._expected_buf_size: int | None = None
        self._closed = False
        if forced_mode is not None:
            self._mode_value.value = int(forced_mode)

    @property
    def transport_mode(self) -> TransportMode:
        if self._mode is TransportMode.UNDECIDED:
            self._mode = TransportMode(self._mode_value.value)
        return self._mode

    @property
    def uses_shared_memory(self) -> bool:
        return self.transport_mode is TransportMode.SHARED_MEMORY

    def _set_mode(self, mode: TransportMode) -> None:
        self._mode = mode
        self._mode_value.value = int(mode)

    def _ensure_mode(self, data: T) -> TransportMode:
        """Choose the data transport based on the first piece of data emitted"""
        if self._mode is not TransportMode.UNDECIDED:
            return self._mode

        if isinstance(data, SMCompliant):
            self._set_mode(TransportMode.SHARED_MEMORY)
        else:
            self._set_mode(TransportMode.QUEUE)
        return self._mode

    def _emit_queue(self, data: T, ts: int) -> bool:
        msg = Message(data, ts)
        success = False

        for q in self._queues:
            try:
                q.put_nowait(msg)
                success = True
            except Full:
                try:
                    q.get_nowait()  # drop oldest
                    q.put_nowait(msg)
                    success = True
                except (Empty, Full):
                    pass  # try next queue

        return success

    def _emit_shared_memory(self, data: SMCompliant, ts: int) -> bool:
        if self._data_type is None:
            self._data_type = type(data)
        elif not isinstance(data, self._data_type):
            raise TypeError(f'Data type mismatch: {type(data)} != {self._data_type}')

        buf_size = data.buf_size()

        if self._sm is None:
            self._expected_buf_size = buf_size
            self._sm = multiprocessing.shared_memory.SharedMemory(create=True, size=buf_size)

            # support multiple receivers
            for sm_q in self._sm_queues:
                metadata = (self._sm.name, buf_size, self._data_type, data.instantiation_params())
                sm_q.put(metadata)
        else:
            assert self._expected_buf_size == buf_size, (
                f'Buffer size mismatch: expected {self._expected_buf_size}, got {buf_size}. '
                'All data instances must have the same buffer size for a given channel.'
            )

        with self._lock:
            data.set_to_buffer(self._sm.buf)
            self._ts_value.value = ts
            for up_value in self._up_values:
                up_value.value = True

        return True

    def emit(self, data: T, ts: int = -1):
        ts = ts if ts >= 0 else self._clock.now_ns()
        mode = self._ensure_mode(data)

        if mode is TransportMode.SHARED_MEMORY:
            if not isinstance(data, SMCompliant):
                raise TypeError('Shared memory transport selected; data must implement SMCompliant')
            self._emit_shared_memory(data, ts)
            return

        self._emit_queue(data, ts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._sm is not None:
            try:
                self._sm.close()
            except BufferError:
                # Receiver may still hold a view; let GC handle once released.
                pass
            else:
                self._sm.unlink()
            self._sm = None

    def __del__(self):
        # Last-resort cleanup when user code forgets to close the emitter.
        self.close()


class MultiprocessReceiver(SignalReceiver[T]):
    """Signal receiver companion for :class:`MultiprocessEmitter`.

    The receiver lazily initialises shared-memory views when the transport mode
    switches and keeps the last queue message as a fallback. Weak references
    back to the emitter let the receiver clear the emitter's cleanup hook on
    close without introducing cycles or non-picklable state.
    """

    def __init__(
        self,
        queue: mp.Queue,
        mode_value: mp.Value,
        lock: mp.Lock,
        ts_value: mp.Value,
        up_value: mp.Value,
        sm_queue: mp.Queue,
        *,
        forced_mode: TransportMode | None = None,
    ):
        self._queue = queue
        self._mode_value = mode_value
        self._forced_mode = forced_mode
        self._mode = forced_mode or TransportMode.UNDECIDED

        # Shared memory state
        self._lock = lock
        self._ts_value = ts_value
        self._up_value = up_value
        self._sm_queue = sm_queue
        self._sm: multiprocessing.shared_memory.SharedMemory | None = None
        self._out_value: SMCompliant | None = None
        self._readonly_buffer: memoryview | None = None

        self._last_queue_message: Message[T] | None = None
        self._closed = False
        if forced_mode is not None:
            self._mode_value.value = int(forced_mode)

    @property
    def transport_mode(self) -> TransportMode:
        if self._mode is TransportMode.UNDECIDED:
            self._mode = TransportMode(self._mode_value.value)
        return self._mode

    @property
    def uses_shared_memory(self) -> bool:
        return self.transport_mode is TransportMode.SHARED_MEMORY

    def _read_queue(self) -> Message[T] | None:
        try:
            message = self._queue.get_nowait()
        except Empty:
            message = None
        else:
            self._last_queue_message = Message(message.data, message.ts, True)
            if self._mode is TransportMode.UNDECIDED:
                self._mode = TransportMode.QUEUE
            return self._last_queue_message

        if self._last_queue_message is None:
            return None

        return Message(self._last_queue_message.data, self._last_queue_message.ts, False)

    def _ensure_shared_memory_initialized(self) -> bool:
        if self._out_value is not None:
            return True

        try:
            sm_name, buf_size, data_type, instantiation_params = self._sm_queue.get_nowait()
        except Empty:
            return False

        self._sm = multiprocessing.shared_memory.SharedMemory(name=sm_name)

        # Unregister from resource tracker to prevent double-cleanup.
        # The emitter (creator) is responsible for unlinking; the receiver only closes.
        # This prevents "leaked shared_memory" warnings on process shutdown.
        try:
            resource_tracker.unregister(self._sm._name, 'shared_memory')
        except Exception:
            # If unregister fails (e.g., not registered), continue anyway.
            pass

        if self._sm.size < buf_size:
            raise RuntimeError(f'Shared memory buffer size mismatch: expected at least {buf_size}, got {self._sm.size}')

        # macOS may return buffers slightly larger than requested; constrain the view.
        self._readonly_buffer = self._sm.buf.toreadonly()[:buf_size]
        self._out_value = data_type(*instantiation_params)
        return True

    def _read_shared_memory(self) -> Message[T] | None:
        with self._lock:
            if self._ts_value.value == -1:
                return None

        if not self._ensure_shared_memory_initialized():
            return None

        with self._lock:
            if self._ts_value.value == -1:
                return None

            assert self._readonly_buffer is not None
            assert self._out_value is not None
            self._out_value.read_from_buffer(self._readonly_buffer)
            updated = self._up_value.value
            self._up_value.value = False
            return Message(data=self._out_value, ts=self._ts_value.value, updated=updated) # instead of True


    def read(self) -> Message[T] | None:
        mode = self.transport_mode

        if mode is TransportMode.SHARED_MEMORY:
            return self._read_shared_memory()

        message = self._read_queue()
        if message is not None:
            return message

        if mode is TransportMode.UNDECIDED:
            # No data yet; underlying transport still undecided.
            return None
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._readonly_buffer is not None:
            self._readonly_buffer.release()
            self._readonly_buffer = None

        if self._sm is not None:
            self._sm.close()
            self._sm = None

    def __del__(self):
        # Ensure shared-memory buffers are released on GC.
        self.close()


class LocalQueueEmitter(SignalEmitter[T]):
    def __init__(self, queue: deque, clock: Clock):
        """Emitter that allows to emit messages to deque.

        Args:
            queue: (deque) Queue to emit to.
            clock: (Clock) Clock to use for timestamps.
        """
        self._queue = queue
        self._clock = clock

    def emit(self, data: T, ts: int = -1):
        self._queue.append(Message(data, ts if ts >= 0 else self._clock.now_ns()))


class LocalQueueReceiver(SignalReceiver[T]):
    def __init__(self, queue: deque):
        """Reader that allows to read messages from deque.

        Args:
            queue: (deque) Queue to read from.
        """
        self._queue = queue
        self._last_value = None

    def read(self) -> Message[T] | None:
        if len(self._queue) > 0:
            self._last_value = self._queue.popleft()
            if self._last_value is not None:
                self._last_value.updated = True
        elif self._last_value is not None:
            self._last_value.updated = False
        return self._last_value


class EventReceiver(SignalReceiver[bool]):
    def __init__(self, event: EventClass, clock: Clock):
        self._event = event
        self._clock = clock
        self._last_value = None

    def read(self) -> Message[bool] | None:
        value = self._event.is_set()
        updated = self._last_value is None or value != self._last_value
        self._last_value = value
        return Message(data=value, ts=self._clock.now_ns(), updated=updated)


class SystemClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    def now_ns(self) -> int:
        return time.monotonic_ns()


def _bg_wrapper(run_func: ControlLoop, stop_event: EventClass, clock: Clock, name: str):
    try:
        for command in run_func(EventReceiver(stop_event, clock), clock):
            match command:
                case Sleep(seconds):
                    time.sleep(seconds)
                case _:
                    raise ValueError(f'Unknown command: {command}')
    except KeyboardInterrupt:
        # Silently handle KeyboardInterrupt in background processes
        pass
    except Exception:
        print(f'\n{"=" * 60}', file=sys.stderr)
        print(f"ERROR in background process '{name}':", file=sys.stderr)
        print(f'{"=" * 60}', file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f'{"=" * 60}\n', file=sys.stderr)
        logging.error(f'Error in control system {name}:\n{traceback.format_exc()}')
    finally:
        logging.info(f'Stopping background process by {name}')
        stop_event.set()


class World:
    """Utility class to bind and run control loops."""

    def __init__(self, clock: Clock | None = None):
        # TODO: stop_signal should be a shared variable, since we should be able to track if background
        # processes are still running
        self._clock = clock or SystemClock()

        self._stop_event = mp.Event()
        self.background_processes = []
        self._manager = mp.Manager()
        self._cleanup_emitters_readers = []
        self.entered = False
        self._connections = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.entered = False
        logging.info('Stopping background processes...')
        self.request_stop()
        time.sleep(0.1)

        logging.info(f'Waiting for {len(self.background_processes)} background processes to terminate...')
        for process in self.background_processes:
            process.join(timeout=3)
            if process.is_alive():
                logging.warning(f'Process {process.name} (pid {process.pid}) did not respond, terminating...')
                process.terminate()
                process.join(timeout=2)  # Give it a moment to terminate
                if process.is_alive():
                    logging.warning(f'Process {process.name} (pid {process.pid}) still alive, killing...')
                    process.kill()
            logging.info(f'Process {process.name} (pid {process.pid}) finished')
            process.close()

        for emitter, receivers in self._cleanup_emitters_readers:
            [receiver.close() for receiver in (receivers if isinstance(receivers, list) else [receivers])]
            emitter.close()


    def request_stop(self):
        self._stop_event.set()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def should_stop_reader(self) -> SignalReceiver[bool]:
        return EventReceiver(self._stop_event, self._clock)

    def interleave(self, *loops: ControlLoop) -> Iterator[Sleep]:
        """Interleave multiple control loops, scheduling them based on their timing requirements.

        This method runs multiple control loops concurrently by executing the next scheduled
        loop and then yielding the wait time until the next execution should occur.

        Args:
            *loops: Variable number of control loops to interleave

        Yields:
            float: Wait times until the next scheduled execution should occur

        Behavior:
            - All loops start at the same time
            - At each step: execute the next scheduled loop, then yield wait time
            - Loops are scheduled for future execution based on their yielded sleep times
            - When any loop completes (StopIteration), the stop event is set
            - Other loops can check the stop event and exit early if desired
            - The method continues until all loops have completed
            - Number of yields equals number of loop executions
        """
        start = self._clock.now()
        counter = 0
        priority_queue = []

        # Initialize all loops with the same start time and unique counters
        for loop in loops:
            priority_queue.append((start, counter, iter(loop(self.should_stop_reader(), self._clock))))
            counter += 1

        heapq.heapify(priority_queue)

        while priority_queue:
            _, _, loop = heapq.heappop(priority_queue)

            try:
                sleep_time = next(loop).seconds
                heapq.heappush(priority_queue, (self._clock.now() + sleep_time, counter, loop))
                counter += 1

                if priority_queue:  # Yield the wait time until the next execution should occur
                    yield Sleep(max(0, priority_queue[0][0] - self._clock.now()))

            except StopIteration:
                # Don't add the loop back and don't yield after a loop completes - it is done
                self.request_stop()

    def connect(
        self,
        emitter: ControlSystemEmitter[T],
        receiver: ControlSystemReceiver[T],
        *,
        emitter_wrapper: Callable[[SignalEmitter[T]], SignalEmitter[T]] = lambda x: x,
        receiver_wrapper: Callable[[SignalReceiver[T]], SignalReceiver[T]] = lambda x: x,
    ):
        """Declare a logical connection between Emitter and Receiver of two control systems.

        The world inspects the ownership of both endpoints when ``start`` is
        called and chooses an appropriate transport (local queue vs.
        multiprocessing pipe). Each Receiver may only be connected once.

        Args:
            emitter: The control system emitter to connect from
            receiver: The control system receiver to connect to
            emitter_wrapper: Optional function to wrap the underlying SignalEmitter
                           before binding. Defaults to identity function.
            receiver_wrapper: Optional function to wrap the underlying SignalReceiver
                            before binding. Defaults to identity function.

        The wrapper functions allow for transformation or decoration of the
        underlying signal transport mechanisms, such as adding logging,
        filtering, or other middleware functionality.
        """
        assert receiver not in [e[1] for e in self._connections], 'Receiver can be connected only to one Emitter'
        assert isinstance(emitter, ControlSystemEmitter)
        assert isinstance(receiver, ControlSystemReceiver)
        if not isinstance(emitter, FakeEmitter) and not isinstance(receiver, FakeReceiver):
            self._connections.append((emitter, receiver, emitter_wrapper, receiver_wrapper))

    def pair(
        self,
        connector: ControlSystemEmitter | ControlSystemReceiver,
        *,
        emitter_wrapper: Callable[[SignalEmitter[T]], SignalEmitter[T]] = lambda x: x,
        receiver_wrapper: Callable[[SignalReceiver[T]], SignalReceiver[T]] = lambda x: x,
    ):
        """Create the complementary connector for an existing endpoint.

        ``World`` infers whether the peer should live locally or in another
        process by looking at the owning control system of each endpoint. To
        keep that inference consistent, ``pair`` instantiates the opposite
        connector class with the same owner and immediately wires the two via
        :meth:`connect`.

        Args:
            connector: Either side of a control-system connection that needs a
                matching peer.
            emitter_wrapper: Optional callable applied to the transport bound to the
                emitter side before the link is registered.
            receiver_wrapper: Optional callable applied to the transport bound to the
                receiver side before the link is registered.

        Returns:
            The freshly created counterpart (`ControlSystemEmitter` for a
            receiver input or `ControlSystemReceiver` for an emitter output).

        Raises:
            ValueError: If ``connector`` is neither an emitter nor a receiver.
        """
        if isinstance(connector, ControlSystemEmitter):
            # We put the same owner, so that both ends are always either local or remote
            receiver = ControlSystemReceiver(connector.owner)
            self.connect(connector, receiver, emitter_wrapper=emitter_wrapper, receiver_wrapper=receiver_wrapper)
            return receiver
        elif isinstance(connector, ControlSystemReceiver):
            emitter = ControlSystemEmitter(connector.owner)
            self.connect(emitter, connector, emitter_wrapper=receiver_wrapper, receiver_wrapper=emitter_wrapper)
            return emitter
        raise ValueError(f'Unsupported connector type: {type(connector)}.')

    def start(
            self,
            main_process: ControlSystem | list[ControlSystem | None],
            background: ControlSystem | list[ControlSystem | None] | None = None,
    ) -> Iterator[Sleep]:
        """Bind declared connections and launch control systems.

        ``main_process`` control systems are scheduled cooperatively in the
        current process, while ``background`` systems are spawned in separate
        processes. Based on the connection map registered via ``connect`` the
        world wires control system emitters and receivers together using local
        queues or multiprocessing queues. Returns an iterator produced by
        ``interleave`` so callers can drive the cooperative scheduler.
        """
        main_process = main_process if isinstance(main_process, list) else [main_process]
        main_process = [m for m in main_process if m is not None]
        background = background or []
        background = background if isinstance(background, list) else [background]
        background = [b for b in background if b is not None]

        local_cs = set(main_process)
        all_cs = local_cs | set(background)

        system_clock = SystemClock()
        local_connections, mp_connections = [], []
        for emitter, receiver, emitter_wrapper, receiver_wrapper in self._connections:
            if emitter.owner in local_cs and receiver.owner in local_cs:
                local_connections.append((emitter, emitter_wrapper, receiver_wrapper(receiver), receiver.maxsize, None))
            elif emitter.owner not in all_cs:
                raise ValueError(f'Emitter {emitter.owner} is not in any control system')
            elif receiver.owner not in all_cs:
                raise ValueError(f'Receiver {receiver.owner} is not in any control system')
            else:
                clock = None if emitter.owner in local_cs else system_clock
                mp_connections.append((emitter, emitter_wrapper, receiver_wrapper(receiver), receiver.maxsize, clock))

        for emitter, emitter_wrapper, receiver, maxsize, _clock in local_connections:
            kwargs = {'maxsize': maxsize} if maxsize is not None else {}
            em, re = self.local_pipe(**kwargs)
            emitter._bind(emitter_wrapper(em))
            receiver._bind(re)

        # Interprocess connection handling
        grouped_mp_connections = defaultdict(list)
        for emitter, emitter_wrapper, receiver, maxsize, clock in mp_connections:
            grouped_mp_connections[emitter].append((emitter_wrapper, receiver, maxsize, clock))

        for emitter_logical, receivers_logical in grouped_mp_connections.items():
            # When emitter lives in a different process, we use system clock to timestamp messages, otherwise we will
            # have to serialise our local clock to the other process, which is not what we want.
            num_receivers = len(receivers_logical)
            emitter_wrapper, _, maxsize, clock = receivers_logical[0]    # parameters the same for all receivers

            kwargs = {'maxsize': maxsize} if maxsize is not None else {}
            emitter_physical, receivers_physical = self.mp_pipes(clock=clock, num_receivers=num_receivers, **kwargs)

            emitter_logical._bind(emitter_wrapper(emitter_physical))

            if not isinstance(receivers_physical, list):
                receivers_physical = [receivers_physical]

            for (_, logical_receiver, _, _), physical_receiver in zip(
                    receivers_logical, receivers_physical, strict=True):
                logical_receiver._bind(physical_receiver)

        self.start_in_subprocess(*[cs.run for cs in background])
        return self.interleave(*[cs.run for cs in main_process])


    def start_in_subprocess(self, *background_loops: ControlLoop):
        """Starts background control loops. Can be called multiple times for different control loops.

        Use `start` whenever possible, as this method is internal.
        """
        for bg_loop in background_loops:
            if hasattr(bg_loop, '__self__'):
                name = f'{bg_loop.__self__.__class__.__name__}.{bg_loop.__name__}'
            else:
                name = getattr(bg_loop, '__name__', 'anonymous')
            # TODO: now we allow only real clock, change clock to a Emitter?
            p = mp.Process(
                target=_bg_wrapper, args=(bg_loop, self._stop_event, SystemClock(), name), daemon=True, name=name
            )
            p.start()
            self.background_processes.append(p)
            logging.info(f'Started background process {name} (pid {p.pid})')

    def local_pipe(self, maxsize: int = 1) -> tuple[SignalEmitter[T], SignalReceiver[T]]:
        """Create a queue-based communication channel within the same process.

        When possible, use `connect` or `pair` instead, as this method is somewhat internal.
        Args:
            maxsize: (int) Maximum queue size (0 for unlimited). Default is 1.

        Returns:
            Tuple of (emitter, reader) for local communication
        """
        q = deque(maxlen=maxsize)
        return LocalQueueEmitter(q, self._clock), LocalQueueReceiver(q)

    def mp_pipes(
            self,
            maxsize: int = 1,
            clock: Clock | None = None,
            *,
            num_receivers: int = 1,
            transport: TransportMode = TransportMode.UNDECIDED
    ) -> tuple[SignalEmitter[T], SignalReceiver[T] | list[SignalReceiver[T]]]:
        """Create an inter-process channel with optional transport override.

        When possible, use `connect` or `pair` instead, as this method is somewhat internal.

        ``transport`` defaults to ``TransportMode.UNDECIDED`` so the first emitted
        payload decides between queue and shared memory. Passing
        :class:`TransportMode.QUEUE` or :class:`TransportMode.SHARED_MEMORY`
        forces a specific transport upfront.

        Args:
            maxsize: Maximum queue size (0 for unlimited). Default is 1.
            clock: Optional clock override for timestamp generation when the
                emitter lives in another process.
            num_receivers: number of receivers to emit. i.e broadcast if > 1
            transport: Transport override. ``TransportMode.UNDECIDED`` enables
                adaptive selection; ``TransportMode.QUEUE`` or
                ``TransportMode.SHARED_MEMORY`` pins the transport.

        Returns:
            Tuple of (emitter, reader-s) suitable for inter-process communication.
        """
        if transport is TransportMode.SHARED_MEMORY and not self.entered:
            raise AssertionError('Shared memory transport is only available after entering the world context.')

        forced_mode: TransportMode | None
        forced_mode = transport if transport in (TransportMode.QUEUE, TransportMode.SHARED_MEMORY) else None

        message_queues = [self._manager.Queue(maxsize=maxsize) for _ in range(num_receivers)]
        lock = self._manager.Lock()
        ts_value = self._manager.Value('Q', -1)
        up_values = [self._manager.Value('b', False) for _ in range(num_receivers)]
        sm_queues = [self._manager.Queue() for _ in range(num_receivers)]
        initial_mode = forced_mode or TransportMode.UNDECIDED
        mode_value = self._manager.Value('i', int(initial_mode))

        emitter_clock = clock or self._clock
        emitter = MultiprocessEmitter(
            emitter_clock, message_queues, mode_value, lock, ts_value, up_values, sm_queues, forced_mode=forced_mode
        )

        receivers = []
        for m_queue, up_value, sm_queue in zip(message_queues, up_values, sm_queues, strict=True):
            receiver = MultiprocessReceiver(
                m_queue, mode_value, lock, ts_value, up_value, sm_queue, forced_mode=forced_mode
            )
            receivers.append(receiver)

        self._cleanup_emitters_readers.append((emitter, receivers))

        return emitter, receivers if num_receivers > 1 else receivers[0]


