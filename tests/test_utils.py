import pytest
from ironic.utils import map_port, Message, OutputPort, NoValue
from ironic.system import ControlSystem, ironic_system
import asyncio


@pytest.mark.asyncio
async def test_map_port_basic_transform():
    original_port = OutputPort("test_port")

    # Create mapped port that doubles the input
    mapped = map_port(lambda x: x * 2, original_port)

    # Track received values
    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(Message(5, timestamp=1000))
    await original_port.write(Message(10, timestamp=2000))

    assert received_values == [10, 20]


@pytest.mark.asyncio
async def test_map_port_preserves_timestamp():
    original_port = OutputPort("test_port")
    mapped = map_port(lambda x: x * 2, original_port)

    received_messages = []

    async def collector(message):
        received_messages.append(message)

    mapped.subscribe(collector)

    test_timestamp = 12345
    await original_port.write(Message(5, timestamp=test_timestamp))

    assert len(received_messages) == 1
    assert received_messages[0].timestamp == test_timestamp


@pytest.mark.asyncio
async def test_map_port_filters_no_value():
    original_port = OutputPort("test_port")

    def transform(x):
        if x < 0:
            return NoValue
        return x * 2

    mapped = map_port(transform, original_port)

    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(Message(5, timestamp=1000))  # Should be mapped
    await original_port.write(Message(-1, timestamp=2000))  # Should be filtered out
    await original_port.write(Message(3, timestamp=3000))  # Should be mapped

    assert received_values == [10, 6]


@pytest.mark.asyncio
async def test_map_port_async_transform():
    original_port = OutputPort("test_port")

    async def async_double(x):
        await asyncio.sleep(0.01)  # Simulate some async work
        return x * 2

    mapped = map_port(async_double, original_port)

    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(Message(5, timestamp=1000))
    await original_port.write(Message(10, timestamp=2000))

    assert received_values == [10, 20]


@pytest.mark.asyncio
async def test_map_port_async_filter():
    original_port = OutputPort("test_port")

    async def async_filter(x):
        await asyncio.sleep(0.01)  # Simulate some async work
        if x < 0:
            return NoValue
        return x * 2

    mapped = map_port(async_filter, original_port)

    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(Message(5, timestamp=1000))  # Should be mapped
    await original_port.write(Message(-1, timestamp=2000))  # Should be filtered out
    await original_port.write(Message(3, timestamp=3000))  # Should be mapped

    assert received_values == [10, 6]
