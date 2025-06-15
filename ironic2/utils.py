from typing import Callable, Any, Tuple

from ironic2 import NoValueType, SignalReader, SignalEmitter, NoValue, Message


class MapSignalReader(SignalReader):

    def __init__(self, reader: SignalReader, func: Callable[[Any], Any]):
        self.reader = reader
        self.func = func

    def value(self):
        orig_message = self.reader.value()
        if orig_message is NoValue:
            return NoValue
        return Message(self.func(orig_message.data), orig_message.ts)


class MapSignalEmitter(SignalEmitter):

    def __init__(self, emitter: SignalEmitter, func: Callable[[Any], Any]):
        self.emitter = emitter
        self.func = func

    def emit(self, message: Message):
        self.emitter.emit(Message(self.func(message.data), message.ts))


def map(reader: SignalReader | SignalEmitter, func: Callable[[Any], Any]) -> SignalReader | SignalEmitter:
    if isinstance(reader, SignalReader):
        return MapSignalReader(reader, func)
    elif isinstance(reader, SignalEmitter):
        return MapSignalEmitter(reader, func)
    else:
        raise ValueError(f"Invalid reader type: {type(reader)}")


class ValueUpdated(SignalReader):

    def __init__(self, reader: SignalReader):
        self.reader = reader
        self.last_ts = 0

    def value(self) -> Tuple[Message | NoValueType, bool]:
        orig_message = self.reader.value()

        if orig_message is NoValue:
            return NoValue, False

        is_updated = orig_message.ts != self.last_ts
        self.last_ts = orig_message.ts

        return orig_message, is_updated
