"""Implementation of multiprocessing channels."""

import heapq
import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import sys
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from enum import IntEnum
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full
from typing import Any, TypeVar

from .core import (
    Clock,
    ControlLoop,
    ControlSystem,
    ControlSystemEmitter,
    ControlSystemReceiver,
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

    def emit(self, data: T, ts: int = -1) -> bool:
        ts = ts if ts >= 0 else self._clock.now_ns()
        try:
            self._queue.put_nowait(Message(data, ts))
            return True
        except Full:
            # Queue is full, try to remove old message and try again
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(Message(data, ts))
                return True
            except (Empty, Full):
                return False


class BroadcastEmitter(SignalEmitter[T]):
    def __init__(self, emitters: Sequence[SignalEmitter[T]]):
        """Emitter that broadcasts messages to all emmiters.

        Args:
            emitters: (Sequence[SignalEmitter]) Emitters to broadcast to.
        """
        self._emitters = emitters

    def emit(self, data: T, ts: int = -1) -> bool:
        any_failed = False
        for emitter in self._emitters:
            any_failed = any_failed or not emitter.emit(data, ts)
        return not any_failed


class QueueReceiver(SignalReceiver[T]):
    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._last_value = None

    def read(self) -> Message[T] | None:
        try:
            self._last_value = self._queue.get_nowait()
        except Empty:
            pass
        return self._last_value


class MultiprocessEmitter(SignalEmitter[T]):
    """Signal emitter that transparently bridges processes.

    The emitter owns both the queue transport and (when selected) a
    shared-memory buffer. It defers the transport choice until the first payload
    unless ``forced_mode`` pins the decision.

    Weak references link the emitter and its paired receiver so that whichever
    side closes first can tell the other to release shared-memory views before
    unlinking the underlying buffer. Full references would create reference cycles
    and break pickling when ``multiprocessing`` spawns child processes, hence
    the indirection via ``weakref``.
    """

    def __init__(
        self,
        clock: Clock,
        queue: mp.Queue,
        mode_value: mp.Value,
        lock: mp.Lock,
        ts_value: mp.Value,
        sm_queue: mp.Queue,
        *,
        forced_mode: TransportMode | None = None,
    ):
        self._clock = clock
        self._queue = queue
        self._mode_value = mode_value
        self._forced_mode = forced_mode
        self._mode = forced_mode or TransportMode.UNDECIDED

        # Shared memory state
        self._data_type: type[SMCompliant] | None = None
        self._lock = lock
        self._ts_value = ts_value
        self._sm_queue = sm_queue
        self._sm: multiprocessing.shared_memory.SharedMemory | None = None
        self._expected_buf_size: int | None = None
        self._receiver_ref: weakref.ReferenceType[MultiprocessReceiver[Any]] | None = None
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

    def _attach_receiver(self, receiver: 'MultiprocessReceiver[Any]') -> None:
        self._receiver_ref = weakref.ref(receiver)

    def _ensure_mode(self, data: T) -> TransportMode:
        if self._mode is not TransportMode.UNDECIDED:
            return self._mode

        if isinstance(data, SMCompliant):
            self._set_mode(TransportMode.SHARED_MEMORY)
        else:
            self._set_mode(TransportMode.QUEUE)
        return self._mode

    def _emit_queue(self, data: T, ts: int) -> bool:
        try:
            self._queue.put_nowait(Message(data, ts))
            return True
        except Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(Message(data, ts))
                return True
            except (Empty, Full):
                return False

    def _emit_shared_memory(self, data: SMCompliant, ts: int) -> bool:
        if self._data_type is None:
            self._data_type = type(data)
        elif not isinstance(data, self._data_type):
            raise TypeError(f'Data type mismatch: {type(data)} != {self._data_type}')

        buf_size = data.buf_size()

        if self._sm is None:
            self._expected_buf_size = buf_size
            self._sm = multiprocessing.shared_memory.SharedMemory(create=True, size=buf_size)
            self._sm_queue.put((self._sm, self._data_type, data.instantiation_params()))
        else:
            assert self._expected_buf_size == buf_size, (
                f'Buffer size mismatch: expected {self._expected_buf_size}, got {buf_size}. '
                'All data instances must have the same buffer size for a given channel.'
            )

        with self._lock:
            data.set_to_buffer(self._sm.buf)
            self._ts_value.value = ts
        return True

    def emit(self, data: T, ts: int = -1) -> bool:
        ts = ts if ts >= 0 else self._clock.now_ns()
        mode = self._ensure_mode(data)

        if mode is TransportMode.SHARED_MEMORY:
            if not isinstance(data, SMCompliant):
                raise TypeError('Shared memory transport selected; data must implement SMCompliant')
            return self._emit_shared_memory(data, ts)

        return self._emit_queue(data, ts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._receiver_ref is not None:
            receiver = self._receiver_ref()
            if receiver is not None:
                receiver.close()

        if self._sm is not None:
            try:
                self._sm.close()
            except BufferError:
                # Receiver may still hold a view; let GC handle once released.
                pass
            else:
                self._sm.unlink()
            self._sm = None

    def __getstate__(self):
        # Drop weakrefs so multiprocessing can pickle the emitter state.
        state = self.__dict__.copy()
        state['_receiver_ref'] = None
        return state

    def __setstate__(self, state):
        # Recreate weakref slot after unpickling in a child process.
        self.__dict__.update(state)
        self._receiver_ref = None

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
        self._sm_queue = sm_queue
        self._sm: multiprocessing.shared_memory.SharedMemory | None = None
        self._out_value: SMCompliant | None = None
        self._readonly_buffer: memoryview | None = None

        self._last_queue_message: Message[T] | None = None
        self._closed = False
        self._emitter_ref: weakref.ReferenceType[MultiprocessEmitter[Any]] | None = None
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

    def _attach_emitter(self, emitter: 'MultiprocessEmitter[Any]') -> None:
        self._emitter_ref = weakref.ref(emitter)

    def _read_queue(self) -> Message[T] | None:
        try:
            self._last_queue_message = self._queue.get_nowait()
        except Empty:
            pass
        else:
            if self._mode is TransportMode.UNDECIDED:
                self._mode = TransportMode.QUEUE
        return self._last_queue_message

    def _ensure_shared_memory_initialized(self) -> bool:
        if self._out_value is not None:
            return True

        try:
            self._sm, data_type, instantiation_params = self._sm_queue.get_nowait()
        except Empty:
            return False

        self._readonly_buffer = self._sm.buf.toreadonly()
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
            return Message(data=self._out_value, ts=self._ts_value.value)

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

        if self._emitter_ref is not None:
            emitter = self._emitter_ref()
            if emitter is not None:
                emitter._receiver_ref = None

    def __del__(self):
        # Ensure shared-memory buffers are released on GC.
        self.close()

    def __getstate__(self):
        # Weakrefs are not picklable; strip them before multiprocessing serialises us.
        state = self.__dict__.copy()
        state['_emitter_ref'] = None
        return state

    def __setstate__(self, state):
        # Restore weakref slot once deserialised in the target process.
        self.__dict__.update(state)
        self._emitter_ref = None


class LocalQueueEmitter(SignalEmitter[T]):
    def __init__(self, queue: deque, clock: Clock):
        """Emitter that allows to emit messages to deque.

        Args:
            queue: (deque) Queue to emit to.
            clock: (Clock) Clock to use for timestamps.
        """
        self._queue = queue
        self._clock = clock

    def emit(self, data: T, ts: int = -1) -> bool:
        self._queue.append(Message(data, ts if ts >= 0 else self._clock.now_ns()))
        return True


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

        return self._last_value


class EventReceiver(SignalReceiver[bool]):
    def __init__(self, event: EventClass, clock: Clock):
        self._event = event
        self._clock = clock

    def read(self) -> Message[bool] | None:
        return Message(data=self._event.is_set(), ts=self._clock.now_ns())


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
        print(f'Stopping background process by {name}', flush=True)
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
        print('Stopping background processes...', flush=True)
        self.request_stop()
        time.sleep(0.1)

        print(f'Waiting for {len(self.background_processes)} background processes to terminate...', flush=True)
        for process in self.background_processes:
            process.join(timeout=3)
            if process.is_alive():
                print(f'Process {process.name} (pid {process.pid}) did not respond, terminating...', flush=True)
                process.terminate()
                process.join(timeout=2)  # Give it a moment to terminate
                if process.is_alive():
                    print(f'Process {process.name} (pid {process.pid}) still alive, killing...', flush=True)
                    process.kill()
            print(f'Process {process.name} (pid {process.pid}) finished', flush=True)
            process.close()

        for emitter, reader in self._cleanup_emitters_readers:
            reader.close()
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
        self._connections.append((emitter, receiver, emitter_wrapper, receiver_wrapper))

    def pair(self, connector: ControlSystemEmitter | ControlSystemReceiver, *, wrapper=lambda x: x):
        """Create the complementary connector for an existing endpoint.

        ``World`` infers whether the peer should live locally or in another
        process by looking at the owning control system of each endpoint. To
        keep that inference consistent, ``pair`` instantiates the opposite
        connector class with the same owner and immediately wires the two via
        :meth:`connect`.

        Args:
            connector: Either side of a control-system connection that needs a
                matching peer.
            wrapper: Optional callable applied to the transport bound to the
                provided ``connector`` before the link is registered.

        Returns:
            The freshly created counterpart (`ControlSystemEmitter` for a
            receiver input or `ControlSystemReceiver` for an emitter output).

        Raises:
            ValueError: If ``connector`` is neither an emitter nor a receiver.
        """
        if isinstance(connector, ControlSystemEmitter):
            # We put the same owner, so that both ends are always either local or remote
            receiver = ControlSystemReceiver(connector.owner)
            self.connect(connector, receiver, emitter_wrapper=wrapper)
            return receiver
        elif isinstance(connector, ControlSystemReceiver):
            emitter = ControlSystemEmitter(connector.owner)
            self.connect(emitter, connector, receiver_wrapper=wrapper)
            return emitter
        raise ValueError(
            f'Unsupported connector type: {type(connector)}. Expected ControlSystemEmitter or ControlSystemReceiver.'
        )

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

        local_connections, mp_connections = [], []
        for emitter, receiver, emitter_wrapper, receiver_wrapper in self._connections:
            if emitter.owner in local_cs and receiver.owner in local_cs:
                local_connections.append((emitter_wrapper(emitter), receiver_wrapper(receiver), receiver.maxsize))
            elif emitter.owner not in all_cs:
                raise ValueError(f'Emitter {emitter.owner} is not in any control system')
            elif receiver.owner not in all_cs:
                raise ValueError(f'Receiver {receiver.owner} is not in any control system')
            else:
                mp_connections.append((emitter_wrapper(emitter), receiver_wrapper(receiver), receiver.maxsize))

        for emitter, receiver, maxsize in local_connections:
            kwargs = {'maxsize': maxsize} if maxsize is not None else {}
            em, re = self.local_pipe(**kwargs)
            emitter._bind(em)
            receiver._bind(re)

        system_clock = SystemClock()
        for emitter, receiver, maxsize in mp_connections:
            # When emitter lives in a different process, we use system clock to timestamp messages, otherwise we will
            # have to serialise our local clock to the other process, which is not what we want.
            clock = None if emitter.owner in local_cs else system_clock
            kwargs = {'maxsize': maxsize} if maxsize is not None else {}
            em, re = self.mp_pipe(clock=clock, **kwargs)
            emitter._bind(em)
            receiver._bind(re)

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
            print(f'Started background process {name} (pid {p.pid})', flush=True)

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

    def mp_pipe(
        self, maxsize: int = 1, clock: Clock | None = None, *, transport: TransportMode = TransportMode.UNDECIDED
    ) -> tuple[SignalEmitter[T], SignalReceiver[T]]:
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
            transport: Transport override. ``TransportMode.UNDECIDED`` enables
                adaptive selection; ``TransportMode.QUEUE`` or
                ``TransportMode.SHARED_MEMORY`` pins the transport.

        Returns:
            Tuple of (emitter, reader) suitable for inter-process communication.
        """
        if transport is TransportMode.SHARED_MEMORY and not self.entered:
            raise AssertionError('Shared memory transport is only available after entering the world context.')

        forced_mode: TransportMode | None
        forced_mode = transport if transport in (TransportMode.QUEUE, TransportMode.SHARED_MEMORY) else None

        message_queue = self._manager.Queue(maxsize=maxsize)
        lock = self._manager.Lock()
        ts_value = self._manager.Value('Q', -1)
        sm_queue = self._manager.Queue()
        initial_mode = forced_mode or TransportMode.UNDECIDED
        mode_value = self._manager.Value('i', int(initial_mode))

        emitter_clock = clock or self._clock
        emitter = MultiprocessEmitter(
            emitter_clock, message_queue, mode_value, lock, ts_value, sm_queue, forced_mode=forced_mode
        )
        receiver = MultiprocessReceiver(message_queue, mode_value, lock, ts_value, sm_queue, forced_mode=forced_mode)
        emitter._attach_receiver(receiver)
        receiver._attach_emitter(emitter)
        self._cleanup_emitters_readers.append((emitter, receiver))
        return emitter, receiver
