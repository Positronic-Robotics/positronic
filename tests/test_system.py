import pytest
from ironic import ControlSystem, Message, ironic_system, on_message, out_property, OutputPort, NoValue
from ironic.utils import last_value


class MockSystemBase(ControlSystem):
    """Base class for mock systems used in testing"""
    def __init__(self):
        super().__init__()
        self.received_messages = []
        self.config_value = None


@ironic_system(
    input_ports=['sensor'],
    output_ports=['processed'],
    input_props=['config'],
    output_props=['status']
)
class MockSystem(MockSystemBase):
    @on_message('sensor')
    async def handle_sensor(self, message: Message):
        self.received_messages.append(message)
        # Echo message to processed output with timestamp
        await self.outs.processed.write(Message(
            data=f"processed_{message.data}",
            timestamp=message.timestamp
        ))

    @out_property
    async def status(self):
        return Message("ok")


class PropertySystemBase(ControlSystem):
    """Base class for property-based mock systems"""
    def __init__(self):
        super().__init__()
        self._multiplier = 1


@ironic_system(
    input_props=['multiplier'],
    output_props=['value']
)
class PropertySystem(PropertySystemBase):
    @out_property
    async def value(self):
        return self._multiplier * 10, 123

    def set_multiplier(self, value):
        self._multiplier = value


@pytest.mark.asyncio
async def test_output_port_subscription():
    """Test that output port subscriptions work."""
    system = MockSystem()  # Updated class name
    received_messages = []

    # Subscribe to the output port with an async lambda
    async def handler(msg):
        received_messages.append(msg)
    system.outs.processed.subscribe(handler)

    # Send a message through input port handler
    test_message = Message(data="test", timestamp=1)
    await system.handle_sensor(test_message)

    assert len(received_messages) == 1
    assert received_messages[0].data == "processed_test"
    assert received_messages[0].timestamp == 1


@pytest.mark.asyncio
async def test_binding():
    """Test that binding inputs to outputs works."""
    system1 = MockSystem()  # Updated class name
    system2 = MockSystem()  # Updated class name

    # Create a test output port
    test_port = OutputPort("test")

    # Bind system1's sensor input to test port
    system1.bind(sensor=test_port)
    # Bind system2's sensor input to system1's processed output
    system2.bind(sensor=system1.outs.processed)

    # Send message through the test port
    test_message = Message(data="test", timestamp=1)
    await test_port.write(test_message)

    # Check that system1 received the original message
    assert len(system1.received_messages) == 1
    assert system1.received_messages[0].data == "test"
    assert system1.received_messages[0].timestamp == 1

    # Check that system2 received the processed message
    assert len(system2.received_messages) == 1
    assert system2.received_messages[0].data == "processed_test"
    assert system2.received_messages[0].timestamp == 1


def test_invalid_binding():
    """Test that binding invalid inputs raises error."""
    system = MockSystem()  # Updated class name

    with pytest.raises(ValueError, match="Unknown input: invalid_port"):
        system.bind(invalid_port=None)


@pytest.mark.asyncio
async def test_property_binding():
    """Test that property binding works."""
    system1 = PropertySystem()
    system2 = PropertySystem()

    # Bind system2's multiplier to system1's value
    system2.bind(multiplier=system1.value)

    # Get initial value
    value, ts = await system2.ins.multiplier()
    assert value == 10  # default multiplier = 1
    assert ts == 123

    # Change system1's multiplier
    system1._multiplier = 2

    # Get updated value
    value, ts = await system2.ins.multiplier()
    assert value == 20
    assert ts == 123


def test_is_bound_returns_false_before_bind():
    system = MockSystem()

    assert not system.is_bound('sensor')


def test_is_bound_returns_true_after_bind():
    system = MockSystem()
    system2 = MockSystem()
    system.bind(sensor=system2.outs.processed)
    assert system.is_bound('sensor')


@pytest.mark.asyncio
async def test_last_value_returns_no_value_initially():
    """Test that last_value returns NoValue before receiving any messages."""
    port = OutputPort("test")
    prop = last_value(port)

    msg = await prop()
    assert msg.data is NoValue
    assert msg.timestamp is not None  # Should have system clock timestamp


@pytest.mark.asyncio
async def test_last_value_returns_last_received_value():
    """Test that last_value returns the last received value after a message is sent."""
    port = OutputPort("test")
    prop = last_value(port)

    test_message = Message(data="test_value", timestamp=123)
    await port.write(test_message)

    msg = await prop()
    assert msg.data == "test_value"
    assert msg.timestamp == 123


@pytest.mark.asyncio
async def test_last_value_updates_with_new_messages():
    """Test that last_value updates when new messages are received."""
    port = OutputPort("test")
    prop = last_value(port)

    # Send first message
    msg1 = Message(data="value1", timestamp=100)
    await port.write(msg1)

    # Verify first message
    result1 = await prop()
    assert result1.data == "value1"
    assert result1.timestamp == 100

    # Send second message
    msg2 = Message(data="value2", timestamp=200)
    await port.write(msg2)

    # Verify property updated to second message
    result2 = await prop()
    assert result2.data == "value2"
    assert result2.timestamp == 200


@pytest.mark.asyncio
async def test_last_value_in_control_system():
    """Test that last_value works when used in a control system binding."""
    # Create test systems
    source = MockSystem()
    target = MockSystem()

    # Create property from source's processed port
    processed_prop = last_value(source.outs.processed)

    # Bind target's config to the property
    target.bind(config=processed_prop)

    # Send message through source system
    test_message = Message(data="test_data", timestamp=300)
    await source.handle_sensor(test_message)

    # Verify target receives processed value through property
    config_msg = await target.ins.config()
    assert config_msg.data == "processed_test_data"
    assert config_msg.timestamp == 300


def test_output_mappings_contains_ports_and_properties():
    """Test that output_mappings contains both output ports and properties."""
    system = MockSystem()

    mappings = system.output_mappings

    # Check that both the output port and property are present
    assert 'processed' in mappings
    assert 'status' in mappings

    # Verify types
    assert isinstance(mappings['processed'], OutputPort)
    assert callable(mappings['status'])
