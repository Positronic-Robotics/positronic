import asyncio
from collections import namedtuple
import time
import signal
from typing import Any, Callable

from ironic.system import (ControlSystem, Message, OutputPort, State, ironic_system, on_message, out_property,
                           system_clock, NoValue)

Change = namedtuple('Change', ['prev', 'current'])


# TODO: Write tests for this and control system
class PropertyChangeDetector:
    """Detects changes in property values and reports them as Change objects.

    This utility class monitors a property for changes in its value and reports them
    along with the previous value when a change is detected. The first call will always
    report a change.

    Args:
        input_prop (Callable[[], Message]): An async callable that returns a Message
            containing the property value to monitor.

    Attributes:
        last_input: The previous value of the monitored property
        initialized (bool): Whether the detector has received its first value

    Example:
        ```python
        async def temperature_prop():
            return Message(data=get_temperature(), timestamp=time.time_ns())

        detector = PropertyChangeDetector(temperature_prop)

        # Will return a Change object on first call
        change = await detector.get_change()  # Change(prev=None, current=20.5)

        # Will return None if temperature hasn't changed
        change = await detector.get_change()  # None

        # Will return a Change object when temperature changes
        change = await detector.get_change()  # Change(prev=20.5, current=21.0)
        ```
    """

    def __init__(self, input_prop: Callable[[], Message]):
        self.input_prop = input_prop
        self.last_input = None
        self.initialized = False

    async def get_change(self):
        input_message = await self.input_prop()
        if input_message.data != self.last_input or not self.initialized:
            self.last_input = input_message.data
            self.initialized = True
            return Message(Change(self.last_input, input_message.data), input_message.timestamp)
        return None


@ironic_system(input_props=["input"], output_ports=["output"])
class PropertyChangeDetectorSystem(ControlSystem):
    """A control system that monitors a property for changes and emits change events.

    This system wraps a PropertyChangeDetector and integrates it into the control system
    framework. It continuously monitors an input property and emits Change objects through
    its output port whenever the input value changes.

    Input Properties:
        input: The property to monitor for changes

    Output Ports:
        output: Emits Message objects containing Change(prev, current) namedtuples
            when the input property value changes

    Example:
        ```python
        # Create and set up a change detector system
        detector_system = PropertyChangeDetectorSystem()

        # Connect it to a temperature sensor system
        detector_system.bind(
            input=temperature_sensor.outs.temperature
        )

        # Subscribe to change events
        async def on_temperature_change(message):
            change = message.data  # Change(prev=20.5, current=21.0)
            print(f"Temperature changed from {change.prev}°C to {change.current}°C")

        detector_system.outs.output.subscribe(on_temperature_change)

        # Run the system
        await run_gracefully(detector_system, temperature_sensor)
        ```
    """

    def __init__(self):
        super().__init__()
        self._detector = PropertyChangeDetector()

    async def step(self):
        change = await self._detector.get_change()
        if change is not None:
            await self.outs.output.write(change)
        return State.ALIVE


class FPSCounter:
    """Utility class for tracking and reporting frames per second (FPS).

    Counts frames and periodically reports the average FPS over the reporting interval.

    Args:
        prefix (str): Prefix string to use in FPS report messages
        report_every_sec (float): How often to report FPS, in seconds (default: 10.0)
    """

    def __init__(self, prefix: str, report_every_sec: float = 10.0):
        self.prefix = prefix
        self.report_every_sec = report_every_sec
        self.reset()

    def reset(self):
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def report(self):
        fps = self.frame_count / (time.monotonic() - self.last_report_time)
        print(f"{self.prefix}: {fps:.2f} fps")
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def tick(self):
        self.frame_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            self.report()


async def run_gracefully(
        system: ControlSystem,
        after_setup_fn: Callable[[], None] | None = None,
        after_cleanup_fn: Callable[[], None] | None = None,
):
    """Runs a control system with graceful shutdown handling.

    This function manages the lifecycle of a ControlSystem, handling setup, continuous operation,
    and cleanup. It sets up signal handlers to catch interrupts (Ctrl+C) and ensures proper
    cleanup of resources when shutting down.

    Args:
        system (ControlSystem): The control system instance to run
        extra_cleanup_fn (Optional[Callable[[], None]]): Optional callback function to perform
            additional cleanup tasks after the system cleanup
        after_setup_fn (Optional[Callable[[], None]]): Optional callback function to perform
            additional steps between the system setup and the main loop

    Example:
        ```python
        system = MyControlSystem()
        def cleanup():
            print("Performing extra cleanup...")

        await run_gracefully(system, extra_cleanup_fn=cleanup)
        ```
    """
    shutdown_event = asyncio.Event()

    def signal_handler(signal, frame):
        print("Program interrupted by user, exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await system.setup()
        if after_setup_fn is not None:
            after_setup_fn()
        while (await system.step()) == State.ALIVE and not shutdown_event.is_set():
            await asyncio.sleep(0)  # Yield to allow other tasks to run, if any
    finally:
        await system.cleanup()
        print('System cleanup finished')
        if after_cleanup_fn:
            after_cleanup_fn()
            print('Extra cleanup finished')


def map_property(function: Callable[[Any], Any], property: Callable[[], Message]):
    """Creates a new property that transforms the data of another property using a mapping function.

    This utility is useful for connecting systems that expect different data formats. It preserves
    the timestamp of the original message while transforming its data content.

    Args:
        function: A function that transforms the data from one format to another
        property: The source property function that returns Messages

    Returns:
        An async function that returns a Message with transformed data and original timestamp

    Example:
        ```python
        # Connect two systems with different position formats
        robot.bind(
            target_position=map_property(
                lambda pos: transform_coordinates(pos),
                inference.outs.target_robot_position
            )
        )
        ```
    """

    @out_property
    async def mapped_property():
        original_message = await property()
        return Message(data=function(original_message.data), timestamp=original_message.timestamp)

    return mapped_property


@ironic_system(input_props=['input'], output_props=['output'])
class MapPropertyCS(ControlSystem):
    """A control system that maps the data of an input property to an output property using a transform function.

    In most cases, you should prefer using `map_property` over this class, as it provides a simpler
    interface for mapping properties. But when you want to input the mapped property in composition, you
    will have to use this class.
    """

    def __init__(self, transform: Callable[[Any], Any]):
        super().__init__()
        self._transform = transform

    @out_property
    async def output(self):
        data = await self.ins.input()
        return Message(data=self._transform(data.data), timestamp=data.timestamp)


@ironic_system(input_ports=['input'], output_ports=['output'])
class MapPortCS(ControlSystem):
    """A control system that maps the data of an input port to an output port using a transform function.

    In most cases, you should prefer using `map_port` over this class, as it provides a simpler
    interface for mapping ports. But when you want to output the mapped port in composition, you
    should use this class.

    Args:
        transform: A function that transforms the data from one format to another
    """

    def __init__(self, transform: Callable[[Any], Any]):
        super().__init__()
        self._transform = transform

    @on_message('input')
    async def handle_input(self, message: Message):
        transformed = self._transform(message.data)
        await self.outs.output.write(Message(transformed, timestamp=message.timestamp))


def map_port(function: Callable[[Any], Any], port: OutputPort) -> OutputPort:
    """Creates a new port that transforms the data of another port using a mapping function.

    This utility creates a new output port that receives messages from the source port,
    transforms their data using the provided function, and emits the transformed messages.
    The timestamp of the original message is preserved.

    The mapping function can be either synchronous or asynchronous (coroutine).

    If the mapping function returns NoValue, the message will be filtered out and not emitted
    on the mapped port. This allows the mapping function to selectively filter messages.

    Args:
        function: A function that transforms the data from one format to another.
                 Can be sync or async. If it returns NoValue, the message will be filtered out.
        port: The source output port to transform messages from

    Returns:
        A new OutputPort that emits the transformed messages

    Example:
        ```python
        # Synchronous transformation
        doubled_port = map_port(lambda x: x * 2, source_port)

        # Asynchronous transformation
        async def async_transform(x):
            await asyncio.sleep(0.1)  # Some async operation
            return x * 2
        async_port = map_port(async_transform, source_port)

        # Filtering with NoValue
        def filter_transform(x):
            if x < 0:  # Filter out negative values
                return NoValue
            return x * 2
        filtered_port = map_port(filter_transform, source_port)
        ```
    """
    fn_name = getattr(function, '__name__', 'mapped_port')
    mapped_port = OutputPort(f"{port.name}_{fn_name}", port.parent_system)

    async def handler(message: Message):
        if asyncio.iscoroutinefunction(function):
            transformed_data = await function(message.data)
        else:
            transformed_data = function(message.data)

        if transformed_data is not NoValue:
            await mapped_port.write(Message(transformed_data, timestamp=message.timestamp))

    port.subscribe(handler)
    return mapped_port


def properties_dict(**properties):
    """Creates a property that returns a dictionary of multiple property values.

    Args:
        **properties: Keyword arguments mapping property names to property functions
            that return Messages

    Returns:
        An async function that returns a Message containing a dictionary of property values.
        The timestamp of the output message is set to the minimum timestamp among all input
        messages.

    Note:
        If the timestamps of the input messages span more than 100ms, a warning will be
        printed since this may indicate synchronization issues between properties.
    """

    async def result():
        # Gather all property values concurrently

        def try_prop_fn(name, prop_fn):
            # improve readability for wrongly connected properties
            try:
                return prop_fn()
            except Exception as e:
                print(f"Error in property {name}: {e}")
                raise e

        messages = await asyncio.gather(*(try_prop_fn(name, prop_fn) for name, prop_fn in properties.items()))

        # Build the dictionaries
        prop_values = {name: messages[i].data for i, name in enumerate(properties.keys())}
        timestamps = [msg.timestamp for msg in messages]

        # Warn if time range is too large
        if timestamps:
            time_range = (max(timestamps) - min(timestamps)) / 1e6  # Convert ns to ms
            if time_range > 100:
                # TODO: Refactor to proper logging, so it could be suppressed
                print(f"Warning: time range for property values is {time_range:.1f} ms. {prop_values.keys()}")

        return Message(data=prop_values, timestamp=min(timestamps))

    return result


def fps_counter(prefix: str, report_every_sec: float = 10.0):
    """
    A decorator that prints the FPS of the decorated function.

    Args:
        prefix: prefix for the FPS report
        report_every_sec: time in seconds between reports
    Example:
        >>> @fps_counter("render")
        >>> def render(self):
        >>>     pass
    """

    def decorator(fn):
        fps_counter = FPSCounter(prefix, report_every_sec)

        def wrapper(*args, **kwargs):
            fps_counter.tick()
            return fn(*args, **kwargs)

        return wrapper

    return decorator


class Throttler:

    def __init__(self, every_sec: float):
        """
        A callable that returns the number of times the function should be called since the last check.
        Args:
            interval: time in seconds between calls
            time_fn: function to get the current time
        """
        self.interval = every_sec
        self.last_time_checked = None

    def __call__(self) -> int:
        """
        Returns the number of times the function should be called since the last check.
        """
        current_time = self.time_fn()

        if self.last_time_checked is None:
            self.last_time_checked = current_time
            return 1

        num_calls = int((current_time - self.last_time_checked) / self.interval)
        if num_calls > 0:
            self.last_time_checked = current_time

        return num_calls

    def time_fn(self) -> float:
        return time.time()


class ThrottledCallback:
    """A callable that executes a callback function at a specified frequency.

    The callback will only be executed if enough time has elapsed since the last call,
    based on the provided frequency in Hz.
    """
    def __init__(self, callback: Callable[[], None], frequency_hz: float):
        self.callback = callback
        self.throttler = Throttler(1.0 / frequency_hz)

    def __call__(self):
        for _ in range(self.throttler()):
            self.callback()


def last_value(port: OutputPort, initial_value: Any = NoValue) -> Callable[[], Message]:
    """Creates a property that returns the last value received from an output port.

    This utility function creates a property that caches and returns the last value
    received from the given output port. The property will return None until the first
    value is received.

    Args:
        port: The output port to monitor

    Returns:
        An async function that returns a Message with the last received value

    Example:
        ```python
        # Create a property that returns the last value from a port
        system.bind(
            last_position=port_to_property(robot.outs.position)
        )
        ```
    """
    last_value = initial_value
    last_timestamp = None

    async def handler(message: Message):
        nonlocal last_value, last_timestamp
        last_value = message.data
        last_timestamp = message.timestamp

    port.subscribe(handler)

    @out_property
    async def property():
        if last_timestamp is None:
            return Message(data=initial_value, timestamp=system_clock())

        return Message(data=last_value, timestamp=last_timestamp)

    return property


@ironic_system(output_ports=['pulse'])
class Pulse(ControlSystem):
    """A control system that emits a pulse at a specified frequency.

    This system emits pulses at a specified frequency. If the system falls behind schedule,
    it will emit multiple pulses to catch up while maintaining the correct intervals.

    Args:
        frequency_hz: Target frequency of pulses
    """

    def __init__(self, frequency_hz: float):
        super().__init__()
        self._period = 1.0 / frequency_hz
        self._next_pulse_time = None

    async def setup(self):
        self._next_pulse_time = time.monotonic()

    async def step(self):
        now = time.monotonic()
        # Emit pulses until we're caught up with the schedule
        while now >= self._next_pulse_time:
            await self.outs.pulse.write(Message(data=None))
            self._next_pulse_time += self._period

        return State.ALIVE


def print_port(port: OutputPort, message_prefix: str = '') -> OutputPort:
    """
    A utility function that prints messages from a output port, keeping the original message.

    Args:
        port: (OutputPort) The output port to print messages from
        message_prefix: (str) A prefix to add to the printed message

    Returns:
        (OutputPort) Mapped output port that prints messages.
    """
    def _print_and_return(message: Message):
        print(f'{message_prefix}: {message}')
        return message

    return map_port(_print_and_return, port)


def const_property(value: Any):
    """Creates a property that returns a constant value.

    This utility function creates a property that returns a constant value.
    """
    @out_property
    async def property():
        return Message(value)

    return property
