import asyncio

import pytest

import ironic as ir


@pytest.mark.asyncio
async def test_map_port_basic_transform():
    original_port = ir.OutputPort("test_port")

    # Create mapped port that doubles the input
    mapped = ir.utils.map_port(lambda x: x * 2, original_port)

    # Track received values
    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(ir.Message(5, timestamp=1000))
    await original_port.write(ir.Message(10, timestamp=2000))

    assert received_values == [10, 20]


@pytest.mark.asyncio
async def test_map_port_preserves_timestamp():
    original_port = ir.OutputPort("test_port")
    mapped = ir.utils.map_port(lambda x: x * 2, original_port)

    received_messages = []

    async def collector(message):
        received_messages.append(message)

    mapped.subscribe(collector)

    test_timestamp = 12345
    await original_port.write(ir.Message(5, timestamp=test_timestamp))

    assert len(received_messages) == 1
    assert received_messages[0].timestamp == test_timestamp


@pytest.mark.asyncio
async def test_map_port_filters_no_value():
    original_port = ir.OutputPort("test_port")

    def transform(x):
        if x < 0:
            return ir.NoValue
        return x * 2

    mapped = ir.utils.map_port(transform, original_port)

    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(ir.Message(5, timestamp=1000))  # Should be mapped
    await original_port.write(ir.Message(-1, timestamp=2000))  # Should be filtered out
    await original_port.write(ir.Message(3, timestamp=3000))  # Should be mapped

    assert received_values == [10, 6]


@pytest.mark.asyncio
async def test_map_port_async_transform():
    original_port = ir.OutputPort("test_port")

    async def async_double(x):
        await asyncio.sleep(0.01)  # Simulate some async work
        return x * 2

    mapped = ir.utils.map_port(async_double, original_port)

    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(ir.Message(5, timestamp=1000))
    await original_port.write(ir.Message(10, timestamp=2000))

    assert received_values == [10, 20]


@pytest.mark.asyncio
async def test_map_port_async_filter():
    original_port = ir.OutputPort("test_port")

    async def async_filter(x):
        await asyncio.sleep(0.01)  # Simulate some async work
        if x < 0:
            return ir.NoValue
        return x * 2

    mapped = ir.utils.map_port(async_filter, original_port)

    received_values = []

    async def collector(message):
        received_values.append(message.data)

    mapped.subscribe(collector)

    # Test sending values
    await original_port.write(ir.Message(5, timestamp=1000))  # Should be mapped
    await original_port.write(ir.Message(-1, timestamp=2000))  # Should be filtered out
    await original_port.write(ir.Message(3, timestamp=3000))  # Should be mapped

    assert received_values == [10, 6]


@pytest.mark.asyncio
async def test_print_port(capfd):
    original_port = ir.OutputPort("test_port")
    print_port = ir.utils.print_port(original_port, "test_port")

    async def collector(message):
        pass

    print_port.subscribe(collector)

    await original_port.write(ir.Message('first msg', timestamp=1000))
    await original_port.write(ir.Message('second msg', timestamp=2000))

    out, err = capfd.readouterr()
    assert out == 'test_port: first msg\ntest_port: second msg\n'
    assert err == ''


@pytest.mark.asyncio
@pytest.mark.parametrize("test_value", [
    42,  # simple integer
    {"key": "value"},  # dictionary
    [1, 2, 3],  # list
    (4, 5, 6),  # tuple
    "test string",  # string
    None,  # None
    {"complex": [1, 2, 3], "nested": {"data": True}},  # nested structure
])
async def test_const_property(test_value):
    prop = ir.utils.const_property(test_value)

    # Test multiple calls to ensure consistency
    message1 = await prop()
    message2 = await prop()

    # Verify the data is correct
    assert message1.data == test_value
    assert message2.data == test_value

    # Verify we get Message instances
    assert isinstance(message1, ir.Message)
    assert isinstance(message2, ir.Message)
