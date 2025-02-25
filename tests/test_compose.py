import pytest
from typing import List, Tuple

import ironic as ir
from ironic.compose import ComposedSystem, CompositionError, compose, extend


@ir.ironic_system(output_ports=['data'], output_props=['counter'])
class DataSource(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self._counter = 0

    @ir.out_property
    async def counter(self):
        return ir.Message(self._counter)

    async def step(self):
        await self.outs.data.write(ir.Message(self._counter))
        self._counter += 1
        return ir.State.ALIVE


@ir.ironic_system(
    input_ports=['in_data'],
    input_props=['counter'],
    output_ports=['processed'],
    output_props=['counter_accumulated'],
)
class Processor(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.received: List[int] = []
        self._counter_accumulated = 0

    @ir.out_property
    async def counter_accumulated(self):
        return ir.Message(self._counter_accumulated)

    @ir.on_message('in_data')
    async def on_data(self, message: ir.Message):
        counter_msg = await self.ins.counter()
        self._counter_accumulated += counter_msg.data

        self.received.append(message.data)
        await self.outs.processed.write(ir.Message(message.data * 2))


@ir.ironic_system(input_ports=['data'], input_props=['counter_accumulated'])
class DataSink(ir.ControlSystem):
    def __init__(self):
        super().__init__()
        self.received: List[Tuple[int, int]] = []

    @ir.on_message('data')
    async def on_data(self, message: ir.Message):
        counter_msg = await self.ins.counter_accumulated()
        self.received.append((message.data, counter_msg.data))


@pytest.mark.asyncio
async def test_simple_composition():
    """Test composing systems with internal connections"""
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    # Compose all systems
    system = compose(
        source,
        processor.bind(counter=source.outs.counter, in_data=source.outs.data),
        sink.bind(
            data=processor.outs.processed,
            counter_accumulated=processor.outs.counter_accumulated,
        ),
    )

    await system.setup()

    # Run for a few steps
    for _ in range(3):
        await system.step()

    await system.cleanup()

    # Check that data flowed through all components
    assert processor.received == [0, 1, 2]
    assert sink.received == [
        (0, 0),
        (2, 1),
        (4, 3),
    ]  # Doubled values and accumulated counter


@pytest.mark.asyncio
async def test_nested_composition():
    """Test composing systems with nested composition"""
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    composed_lvl1 = compose(
        source,
        processor.bind(counter=source.outs.counter, in_data=source.outs.data),
        outputs={
            'result': processor.outs.processed,
            'counter': processor.outs.counter_accumulated,
        },
    )

    composed_lvl2 = compose(
        composed_lvl1,
        sink.bind(
            data=composed_lvl1.outs.result,
            counter_accumulated=composed_lvl1.outs.counter,
        ),
        outputs={
            'result': composed_lvl1.outs.result,
            'counter': composed_lvl1.outs.counter,
        },
    )

    await composed_lvl2.setup()

    # Run for a few steps
    for _ in range(3):
        await composed_lvl2.step()

    await composed_lvl2.cleanup()

    # Check that data flowed through all components
    assert processor.received == [0, 1, 2]
    assert sink.received == [
        (0, 0),
        (2, 1),
        (4, 3),
    ]  # Doubled values and accumulated counter


@pytest.mark.asyncio
async def test_composition_with_not_included_components_produces_exception():
    not_included_source = DataSource()
    processor = Processor()
    sink = DataSink()

    with pytest.raises(CompositionError):
        compose(
            processor.bind(
                counter=not_included_source.outs.counter,
                in_data=not_included_source.outs.data,
            ),
            sink.bind(
                data=processor.outs.processed,
                counter_accumulated=processor.outs.counter_accumulated,
            ),
        )


@pytest.mark.asyncio
async def test_composition_with_not_included_input_component_produces_exception():
    processor_1 = Processor()
    processor_2 = Processor()

    with pytest.raises(
        CompositionError, match="Input mappings reference components not in composition"
    ):
        compose(
            processor_1,
            inputs={
                'data': (processor_2, 'in_data'),
                'counter': (processor_2, 'counter'),
            },
        )


@pytest.mark.asyncio
async def test_composition_with_not_included_output_component_produces_exception():
    source = DataSource()
    processor_1 = Processor()
    processor_2 = Processor()

    processor_2.bind(
        in_data=processor_1.outs.processed, counter=processor_1.outs.counter_accumulated
    )

    with pytest.raises(
        CompositionError,
        match="Output mappings reference components not in composition",
    ):
        compose(
            source,
            processor_1.bind(counter=source.outs.counter, in_data=source.outs.data),
            outputs={
                'result': processor_2.outs.processed,
                'counter': processor_2.outs.counter_accumulated,
            },
        )


@pytest.mark.asyncio
async def test_exposed_outputs():
    source = DataSource()
    processor = Processor()

    # Compose with exposed output
    system = compose(
        source,
        processor.bind(counter=source.counter, in_data=source.outs.data),
        outputs={
            'result': processor.outs.processed,
            'counter': processor.outs.counter_accumulated,
        },
    )

    # Create external sink
    sink = DataSink().bind(
        data=system.outs.result, counter_accumulated=system.outs.counter
    )

    await system.setup()
    await sink.setup()

    # Run for a few steps
    for _ in range(3):
        await system.step()
        await sink.step()

    await sink.cleanup()
    await system.cleanup()

    assert processor.received == [0, 1, 2]
    assert sink.received == [(0, 0), (2, 1), (4, 3)]


@pytest.mark.asyncio
async def test_exposed_inputs():
    processor = Processor()
    sink = DataSink()

    # Compose with exposed input
    system = compose(
        processor,
        sink.bind(
            data=processor.outs.processed,
            counter_accumulated=processor.outs.counter_accumulated,
        ),
        inputs={'data': (processor, 'in_data'), 'counter': (processor, 'counter')},
    )

    # Create external source
    source = DataSource()

    # Connect composed system to source
    system.bind(data=source.outs.data, counter=source.outs.counter)

    await source.setup()
    await system.setup()

    # Run for a few steps
    for _ in range(3):
        await source.step()
        await system.step()

    await source.cleanup()
    await system.cleanup()

    assert processor.received == [0, 1, 2]
    assert sink.received == [(0, 0), (2, 1), (4, 3)]


@pytest.mark.asyncio
async def test_invalid_input():
    """Test error handling for invalid inputs"""
    processor = Processor()
    system = compose(processor, inputs={'valid_input': (processor, 'in_data')})

    with pytest.raises(ValueError, match="Unknown input: invalid"):
        system.bind(invalid=None)


@pytest.mark.asyncio
async def test_invalid_component_references():
    """Test validation of component references in inputs/outputs"""
    source = DataSource()
    processor = Processor()
    other_processor = Processor()  # Not included in composition

    # Test invalid input reference
    with pytest.raises(
        CompositionError, match="Input mappings reference components not in composition"
    ):
        compose(
            source,
            processor,
            inputs={
                'data': (other_processor, 'in_data')
            },  # other_processor not in components
        )

    # Test invalid output reference
    with pytest.raises(
        CompositionError,
        match="Output mappings reference components not in composition",
    ):
        compose(
            source,
            processor,
            outputs={
                'result': other_processor.outs.processed
            },  # other_processor not in components
        )

    # Valid composition should work
    system = compose(
        source,
        processor,
        inputs={'data': (processor, 'in_data')},
        outputs={'result': processor.outs.processed},
    )
    assert isinstance(system, ComposedSystem)


@pytest.mark.asyncio
async def test_extend_system_appends_output():
    """Test extending a control system with additional outputs"""
    source = DataSource()

    source = extend(
        source,
        {'counter_squared': ir.utils.map_property(lambda x: x**2, source.outs.counter)},
    )

    await source.setup()
    value = await source.outs.counter_squared()
    assert value.data == 0

    await source.step()
    value = await source.outs.counter_squared()
    assert value.data == 1

    await source.step()
    value = await source.outs.counter_squared()
    assert value.data == 4

    await source.step()
    value = await source.outs.counter_squared()
    assert value.data == 9

    await source.cleanup()


@pytest.mark.asyncio
async def test_extend_composed_system_appends_output():
    """Test extending a composed system with additional outputs"""
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    composed = compose(
        source,
        processor.bind(counter=source.outs.counter, in_data=source.outs.data),
        sink.bind(
            data=processor.outs.processed,
            counter_accumulated=processor.outs.counter_accumulated,
        ),
        outputs={
            'result': processor.outs.processed,
            'counter_accumulated': processor.outs.counter_accumulated,
        },
    )

    extended = extend(
        composed,
        {
            'counter_squared': ir.utils.map_property(
                lambda x: x**2, composed.outs.counter_accumulated
            )
        },
    )

    await extended.setup()

    await extended.step()
    value = await extended.outs.counter_squared()
    assert value.data == 0

    await extended.step()
    value = await extended.outs.counter_squared()
    assert value.data == 1

    await extended.step()
    value = await extended.outs.counter_squared()
    assert value.data == 9  # (0 + 1 + 2) ** 2

    await extended.step()
    value = await extended.outs.counter_squared()
    assert value.data == 36  # (0 + 1 + 2 + 3) ** 2

    await extended.step()
    value = await extended.outs.counter_squared()
    assert value.data == 100  # (0 + 1 + 2 + 3 + 4) ** 2

    await extended.cleanup()


@pytest.mark.asyncio
async def test_extend_composed_system_inputs_could_be_bound_after_extension():
    """Test extending a composed system with additional outputs"""
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    composed = compose(
        processor,
        sink.bind(
            data=processor.outs.processed,
            counter_accumulated=processor.outs.counter_accumulated,
        ),
        inputs={'data': (processor, 'in_data'), 'counter': (processor, 'counter')},
        outputs={
            'result': processor.outs.processed,
            'counter_accumulated': processor.outs.counter_accumulated,
        },
    )

    extended = extend(
        composed,
        {
            'counter_squared': ir.utils.map_property(
                lambda x: x**2, composed.outs.counter_accumulated
            )
        },
    )

    source_with_extended = compose(
        source,
        extended.bind(data=source.outs.data, counter=source.outs.counter),
        outputs={'counter_squared': extended.outs.counter_squared},
    )

    await source_with_extended.setup()

    await source_with_extended.step()
    value = await source_with_extended.outs.counter_squared()
    assert value.data == 0

    await source_with_extended.step()
    value = await source_with_extended.outs.counter_squared()
    assert value.data == 1

    await source_with_extended.step()
    value = await source_with_extended.outs.counter_squared()
    assert value.data == 9  # (0 + 1 + 2) ** 2


@pytest.mark.asyncio
async def test_compose_system_appears_twice_in_composition_produces_exception():
    source = DataSource()

    with pytest.raises(CompositionError, match=".*DataSource.*"):
        compose(source, source)


@pytest.mark.asyncio
async def test_compose_system_appears_twice_in_subsystem_composition_produces_exception():
    source = DataSource()
    processor = Processor()
    sink = DataSink()

    composed = compose(
        source,
        processor,
        sink.bind(
            data=processor.outs.processed,
            counter_accumulated=processor.outs.counter_accumulated,
        ),
        outputs={
            'result': processor.outs.processed,
            'counter_accumulated': processor.outs.counter_accumulated,
        },
    )

    with pytest.raises(CompositionError, match=".*DataSource.*"):
        compose(source, composed)


@pytest.mark.asyncio
async def test_multiple_input_mappings():
    """Test composing systems with multiple input mappings"""
    source = DataSource()
    processor1 = Processor()
    processor2 = Processor()

    # Compose with multiple input mappings
    system = compose(
        processor1,
        processor2,
        inputs={
            'data': [(processor1, 'in_data'), (processor2, 'in_data')],
            'counter': [(processor1, 'counter'), (processor2, 'counter')],
        },
    )

    # Create external source
    source = DataSource()

    # Connect composed system to source
    system.bind(data=source.outs.data, counter=source.outs.counter)

    await source.setup()
    await system.setup()

    # Run for a few steps
    for _ in range(3):
        await source.step()
        await system.step()

    await source.cleanup()
    await system.cleanup()

    # Both processors should have received the same data
    assert processor1.received == [0, 1, 2]
    assert processor2.received == [0, 1, 2]
