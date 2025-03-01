import asyncio
import inspect
import re
import time
import warnings

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


# Object that represents no value
# It is used to make a distinction between a value that is not set and a value that is set to None
class NoValue:
    def __str__(self):
        return "ironic.NoValue"

    def __repr__(self):
        return "ironic.NoValue"


NoValue = NoValue()


class State(Enum):
    ALIVE = 0
    FINISHED = 1


@dataclass
class Message:
    """
    Contains some data and a timestamp for this data. Timestamps are integers,
    to avoid floating point precision issues. It can be related to epoch or
    to anything else, depending on the context.

    If no timestamp is provided, the current system time is used.
    """
    data: Any
    timestamp: int = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = system_clock()


def system_clock() -> int:
    """Get current timestamp in nanoseconds."""
    return time.monotonic_ns()


class OutputPort:

    def __init__(self, name: str, parent_system: Optional["ControlSystem"] = None):
        self._name = name
        self._handlers = []
        self._parent_system = parent_system

    @property
    def parent_system(self):
        return self._parent_system

    def subscribe(self, handler):
        """Subscribe a handler to this port. Handler must be an async function."""
        self._handlers.append(handler)

    @property
    def name(self):
        return self._name

    async def write(self, message: Message):
        """
        Write a message to the port. Implementations must check that the timestamp
        is in increasing order.
        """
        # We use asyncio.gather and sacrifice reproducibility of results.
        # If you want reproducibility, run handlers sequentially.
        await asyncio.gather(*[handler(message) for handler in self._handlers])


class StubOutputPort(OutputPort):
    """
    A stub output port that does nothing. Useful for ir.compose() when you don't
    want to have an output port in the composition, but don't have a real output for it.
    """

    def __init__(self):
        super().__init__("stub", None)

    async def write(self, message: Message):
        pass


OutputPort.Stub = StubOutputPort


def _validate_pythonic_name(name: str, context: str) -> None:
    """
    Validate that a name follows Python naming conventions.
    Allows letters (both cases), numbers and underscores, must start with a letter.
    """
    pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')
    if not pattern.match(name):
        raise ValueError(f"Invalid {context} name '{name}'. Names must start with a letter "
                         "and contain only letters, numbers, and underscores")


def out_property(method: Callable[..., Awaitable[Message]]) -> Callable[..., Awaitable[Message]]:
    """
    Decorator for declaring that the method is an output property.

    The decorated method must be an async method that returns an `ir.Message`.
    The method name must be listed in output_props of the ironic_system decorator.

    Args:
        method: The async method to decorate

    Returns:
        The decorated method

    Raises:
        ValueError: If the decorated method is not async
    """
    if not inspect.iscoroutinefunction(method):
        raise ValueError(f"Output property '{method.__name__}' must be an async method")

    async def wrapper(*args, **kwargs) -> Message:
        return await method(*args, **kwargs)

    wrapper.__is_output_property__ = True
    wrapper.__output_property_name__ = method.__name__

    return wrapper


def on_message(name: str):
    """
    Decorator for declaring that the method is a callback for a message.

    Args:
        name: The name of the input port this method handles messages from

    The decorated method should have the signature:
        async def method(self, message: Message)

    Raises:
        ValueError: If the decorated method is not async
    """

    def decorator(method):
        if not inspect.iscoroutinefunction(method):
            raise ValueError(f"Message handler '{method.__name__}' for port '{name}' must be an async function")

        # Store the port name this method handles
        method.__is_message_handler__ = True
        method.__input_port_name__ = name
        return method

    return decorator


def ironic_system(*,  # noqa: C901  Function is too complex
                  input_ports: List[str] = None,
                  output_ports: List[str] = None,
                  input_props: List[str] = None,
                  output_props: List[str] = None):
    """
    Class decorator for defining ironic system interfaces.
    """
    input_ports = input_ports or []
    output_ports = output_ports or []
    input_props = input_props or []
    output_props = output_props or []

    # Validate naming conventions
    for port in input_ports:
        _validate_pythonic_name(port, "input port")
    for port in output_ports:
        _validate_pythonic_name(port, "output port")
    for prop in input_props:
        _validate_pythonic_name(prop, "input property")
    for prop in output_props:
        _validate_pythonic_name(prop, "output property")

    # Check for duplicate names across all inputs and outputs
    all_inputs = input_ports + input_props
    duplicate_inputs = [x for x in all_inputs if all_inputs.count(x) > 1]
    if duplicate_inputs:
        raise ValueError(f"Duplicate input names declared: {set(duplicate_inputs)}")

    all_outputs = output_ports + output_props
    duplicate_outputs = [x for x in all_outputs if all_outputs.count(x) > 1]
    if duplicate_outputs:
        raise ValueError(f"Duplicate output names declared: {set(duplicate_outputs)}")

    def decorator(cls):
        cls._input_ports = input_ports
        cls._output_ports = output_ports
        cls._input_props = input_props
        cls._output_props = output_props

        # Warn about input/output name conflicts
        input_output_conflicts = set(all_inputs) & set(all_outputs)
        if input_output_conflicts:
            warnings.warn(
                f"Names used as both input and output: {input_output_conflicts} in class {cls.__name__}. "
                "This might make the system harder to understand.", UserWarning)

        # Verify output properties
        defined_props = {
            method.__output_property_name__
            for method in cls.__dict__.values() if hasattr(method, '__is_output_property__')
        }

        # Check for properties not declared in output_props
        undeclared_props = defined_props - set(output_props)
        if undeclared_props:
            raise ValueError(f"Output properties {undeclared_props} in class {cls.__name__} "
                             "are not listed in output_props of @ironic_system")

        # Check for missing properties
        missing_props = set(output_props) - defined_props
        if missing_props:
            raise ValueError(f"Output properties {missing_props} are not defined in class {cls.__name__}")

        # Verify message handlers (but don't store them in class)
        handlers = {
            method.__input_port_name__: method
            for method in cls.__dict__.values() if hasattr(method, '__is_message_handler__')
        }

        # Check for duplicate handlers
        duplicate_handlers = [x for x in handlers.keys() if list(handlers.keys()).count(x) > 1]
        if duplicate_handlers:
            raise ValueError(f"Multiple handlers defined for input ports {set(duplicate_handlers)} "
                             f"in class {cls.__name__}")

        return cls

    return decorator


class ControlSystem:
    """A control system that can be connected to other systems through ports and properties.

    A control system has input and output ports for push-based communication (similar to channels),
    and input and output properties for pull-based communication (similar to RPC).

    The system can be either:
    - Passive: only reacts to incoming messages on its input ports
    - Active: also performs periodic work in its step() method

    Ports and properties must be declared using the @ironic_system decorator. Input ports can have
    message handlers defined using @on_message, while output properties are defined using
    @output_property.

    Binding must be done before calling setup(), as control system can start writing to output ports
    in setup().
    """

    def __init__(self):
        self._message_handlers = {
            method.__input_port_name__: getattr(self, name)  # Get bound methods instead of functions
            for name, method in self.__class__.__dict__.items() if hasattr(method, '__is_message_handler__')
        }
        self._output_props = {
            method.__output_property_name__: getattr(self, method.__output_property_name__)
            for method in self.__class__.__dict__.values() if hasattr(method, '__is_output_property__')
        }
        outs = {port: OutputPort(port, self) for port in self._output_ports}
        outs.update(self._output_props)
        self.outs = SimpleNamespace(**outs)
        self.ins = None

        self._setup_done = False

    @property
    def output_mappings(self) -> Dict[str, Union[OutputPort, Callable[[Any], Any]]]:
        """
        Get a dictionary of output mappings for the control system. Useful with compose(), when
        the outputs of composed systems equal to the outputs of one of the sub-systems.

        Returns:
            A dictionary of output mappings
        """
        return vars(self.outs)

    def is_bound(self, input_name: str) -> bool:
        """
        Check if the input has been bound to an output.

        Args:
            input_name: (str) The name of the input to check

        Returns:
            (bool) True if the input has been bound to an output, False otherwise
        """
        if self.ins is None:
            return False

        return hasattr(self.ins, input_name)

    def bind(self, **bindings):
        """Bind inputs to provided outputs of other systems. Must be called before calling setup.

        For convienice, returns self so result can be passed to other functions."""
        assert not self._setup_done, "bind() must be called before setup()"

        if self.ins is None:
            self.ins = SimpleNamespace()

        for name, incoming_output in bindings.items():
            if name in self._input_ports:
                if name in self._message_handlers:  # Otherwise we just ignore that input
                    setattr(self.ins, name, incoming_output.subscribe(self._message_handlers[name]))
            elif name in self._input_props:
                if not is_property(incoming_output):
                    raise ValueError(
                        f"{name} must be bound to a property (async function), got {type(incoming_output)}")
                setattr(self.ins, name, incoming_output)  # Property is just a callback, so assignment is enough
            else:
                raise ValueError(f"Unknown input: {name}. Inputs of {self.__class__.__name__}: {self._input_ports}")
        return self

    async def setup(self):
        """Setup the control system."""
        self._setup_done = True

    async def cleanup(self):
        """Cleanup the control system."""
        pass

    async def step(self) -> State:
        """
        Perform periodic work and return the system's state. Every step may be your last,
        so write the code accordingly.

        This method is called repeatedly while the system is running. The returned State
        indicates whether the system should continue running (State.ALIVE) or has finished
        its work (State.FINISHED).

        By default does nothing and returns State.ALIVE. Override this method to implement
        active behavior.

        IMPORTANT: If you run your system with other systems (which is what you are here for),
        the step must be non-blocking and return back as soon as possible. If you really need
        to perform blocking operations, do them in a separate thread or asyncio.run_coroutine_threadsafe.

        Returns:
            State: The current state of the system - either State.ALIVE to continue running
                  or State.FINISHED to signal completion.
        """
        return State.ALIVE


def is_port(obj: Any) -> bool:
    """Check if the object is an output port."""
    return isinstance(obj, OutputPort)


def is_property(obj: Any) -> bool:
    """Check if the object is an output property."""
    return inspect.iscoroutinefunction(obj)
