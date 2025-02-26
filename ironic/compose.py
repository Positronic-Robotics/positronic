"""
Composition utilities for ironic control systems.

Example usage:
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    # Simple composition with internal connections and properties
    system = compose(
        source,
        processor.bind(counter=source.outs.counter,
                      in_data=source.outs.data),
        sink.bind(data=processor.outs.processed,
                 counter_accumulated=processor.outs.counter_accumulated)
    )

    # Or with explicit input/output mapping:
    system = compose(
        components=[processor, sink],
        inputs={'data': (processor, 'in_data')},
        outputs={'result': (processor, 'processed')}
    )

    # Input mappings can include None to skip certain ports or properties
    # Output mappings can include special ir.OutputPort.Stub() to skip certain ports:
    system = compose(
        components=[processor, sink],
        inputs={
            'data': (processor, 'in_data'),
            'optional_input': None  # This input will be skipped during binding
        },
        outputs={
            'result': processor.outs.processed,
            'optional_output': ir.OutputPort.Stub()  # This output will be skipped during connection
        }
    )

    # Multiple input mappings:
    system = compose(
        components=[processor1, processor2],
        inputs={
            'data': [(processor1, 'in_data'), (processor2, 'in_data')]  # One input mapped to multiple inner inputs
        }
    )
"""

import asyncio
from typing import Callable, Dict, List, Sequence, Set, Tuple, Optional, Union, Any
from types import SimpleNamespace

from .system import ControlSystem, OutputPort, ironic_system, State, is_port, is_property


def get_parent_system(obj: Any) -> Optional[ControlSystem]:
    """
    Get the parent system of an object if possible.

    For OutputPort objects, the parent system is the system that owns the port.
    For output properties, which are the methods decorated with @out_property, the parent system is the instance that
    defines the property. For other objects, the parent system is None. That could be an objects created with functions
    like `ironic.utils.map_property`.

    Args:
        obj: The object to get the parent system of

    Returns:
        The parent system of the object, or None if the object is not a ControlSystem
    """
    if obj is None:
        return None

    if isinstance(obj, OutputPort):
        return obj.parent_system

    if hasattr(obj, '__self__'):
        return obj.__self__

    return None


class CompositionError(Exception):
    """Error raised when composition validation fails"""
    pass


def _gather_components(components: Sequence[ControlSystem]) -> Set[ControlSystem]:
    """
    Recursively gather all components from a sequence of control systems.

    Args:
        components: A sequence of control systems to gather components from

    Returns:
        A set of all components in the composition

    Raises:
        CompositionError: If a component appears more than once in the composition
    """
    res = set()

    for component in components:
        if component in res:
            raise CompositionError(f"Component {component} appears more than once in composition.")
        res.add(component)

        if isinstance(component, ComposedSystem):
            subsystems = _gather_components(component._components)
            intersection = subsystems.intersection(res)
            if intersection:
                raise CompositionError(
                    f"Subsystem {component} contains components that are already in the composition: {intersection}")
            res.update(subsystems)

    return res


@ironic_system(input_ports=[], output_ports=[])
class ComposedSystem(ControlSystem):
    """A control system composed of other control systems"""

    def __init__(self,  # noqa: C901  Function is too complex
                 components: Sequence[ControlSystem],
                 inputs: Optional[Dict[str, Union[None, Tuple[ControlSystem, str],
                                                  List[Tuple[ControlSystem, str]]]]] = None,
                 outputs: Optional[Dict[str, OutputPort | Callable]] = None):
        components_set = _gather_components(components)

        # Validate all referenced components exist in composition
        if inputs:
            for input_mapping in inputs.values():
                if input_mapping is None:
                    continue
                if isinstance(input_mapping, tuple):
                    input_mapping = [input_mapping]
                for comp, _ in input_mapping:
                    if comp not in components_set:
                        raise CompositionError("Input mappings reference components not in composition")

        if outputs:
            for name, out in outputs.items():
                # Validate output type
                if not (is_port(out) or is_property(out)):
                    raise CompositionError(
                        f"Output '{name}' must be either an OutputPort or an async function, got {type(out)}")

                parent_system = get_parent_system(out)
                if parent_system is not None and parent_system not in components_set:
                    raise CompositionError(
                        f"Output mappings reference components not in composition. \n"
                        f"  Output: '{name}'. \n"
                        f"  Parent system: {parent_system}. \n"
                        f"  Systems in composition: {components_set} \n")

        self._validate_bound_inputs(components, components_set)

        # Get input/output ports from mappings
        input_ports = list(inputs.keys()) if inputs else []
        output_ports = list(outputs.keys()) if outputs else []

        # Update class-level port definitions
        self.__class__._input_ports = input_ports
        self.__class__._output_ports = output_ports

        super().__init__()
        self._components = components

        # Store the component and port name for each input
        self._input_mappings = dict(inputs or {})

        # Connect outputs - direct assignment of OutputPort objects
        if outputs:
            for name, original_port in outputs.items():
                if original_port is None:
                    continue
                setattr(self.outs, name, original_port)

    async def setup(self):
        """Set up all components"""
        for component in self._components:
            await component.setup()

    async def cleanup(self):
        """Clean up all components"""
        for component in self._components:
            await component.cleanup()

    async def step(self):
        """Step all components"""
        results = await asyncio.gather(*[component.step() for component in self._components])
        return State.ALIVE if all(result == State.ALIVE for result in results) else State.FINISHED

    def bind(self, **bindings):
        """Bind inputs to the appropriate components"""
        binds = {}

        for name, binding in bindings.items():
            if name not in self._input_mappings:
                raise ValueError(f"Unknown input: {name}")

            input_mapping = self._input_mappings[name]
            if input_mapping is None:
                continue

            if isinstance(input_mapping, tuple):
                input_mapping = [input_mapping]

            for component, port_name in input_mapping:
                component.bind(**{port_name: binding})

            binds[name] = binding

        self.ins = SimpleNamespace(**binds)
        return self

    def _validate_bound_inputs(self, components: Sequence[ControlSystem], components_set: Set[ControlSystem]):
        """Validate that all bound inputs reference components in the composition"""
        for c in components:
            if c.ins is not None:
                for name, binding in c.ins.__dict__.items():
                    parent_system = get_parent_system(binding)

                    if parent_system is not None and parent_system not in components_set:
                        raise CompositionError(f"Bound input references component not in composition. \n"
                                               f"  Input: '{name}'. \n"
                                               f"  Component: {parent_system}. \n"
                                               f"  Systems in composition: {components_set} \n")


# TODO: ideally we want inputs to be constructed from the components directly like outputs.
# Figure out the way to do this since we currently don't have any InputPort class.
def compose(*components: ControlSystem,
            inputs: Dict[str, Union[None, Tuple[ControlSystem, str], List[Tuple[ControlSystem, str]]]] = None,
            outputs: Dict[str, OutputPort | Callable] = None) -> ComposedSystem:
    """
    Compose multiple control systems into a single system.

    The composed system exposes selected inputs and outputs that can be bound to other systems.
    This allows hierarchical composition where composed systems can be treated as regular
    control systems and connected to other systems.

    Args:
        *components: Variable number of control system instances to compose
        inputs: Dictionary mapping external input names to either:
               - None (to skip the input)
               - A tuple of (component, port_name) for single mapping
               - A list of tuples [(component, port_name), ...] for multiple mappings
               These inputs can later be bound to outputs of other systems
        outputs: Dictionary mapping external output names to output ports or properties.
                These outputs can be bound to inputs of other systems

    Returns:
        A composed control system instance that can be bound to other systems

    Example:
        source = DataSource()
        processor = Processor()
        sink = DataSink()

        # Simple composition with internal connections
        system = compose(
            source,
            processor.bind(counter=source.outs.counter,
                           in_data=source.outs.data),
            sink.bind(data=processor.outs.processed,
                      counter_accumulated=processor.outs.counter_accumulated)
        )

        # Composition with exposed ports that can be bound to other systems:
        subsystem = compose(
            processor,
            sink,
            inputs={'data': (processor, 'in_data')},
            outputs={'result': processor.outs.processed}
        )

        # Multiple input mappings:
        system = compose(
            source,
            processor1,
            processor2,
            inputs={
                'data': [(processor1, 'in_data'), (processor2, 'in_data')]  # One input mapped to multiple inner inputs
            }
        )
    """
    return ComposedSystem(
        components=list(components),  # Convert tuple to list
        inputs=inputs,
        outputs=outputs)


def extend(system: ControlSystem, outputs: Dict[str, OutputPort | Callable]) -> ComposedSystem:
    """
    Extend a control system with additional outputs.

    This function allows adding new outputs to an existing control system without
    modifying its existing structure. It returns a new composed system that includes
    both the original system and the new outputs.

    Args:
        system: The control system to extend
        outputs: Dictionary mapping additional output names to output ports or properties

    Returns:
        A new composed control system instance that includes both the original system
        and the new outputs

    Example:
        >>> system = DataSource()
        >>> extended = extend(system, {'value_squared': ir.utils.map_property(lambda x: x ** 2, system.outs.value)})
    """
    original_inputs = {}
    for input_port in system._input_ports:
        original_inputs[input_port] = (system, input_port)

    original_outputs = {name: getattr(system.outs, name) for name in system.outs.__dict__.keys()}
    original_outputs.update(outputs)

    return ComposedSystem(components=[system], inputs=original_inputs, outputs=original_outputs)
