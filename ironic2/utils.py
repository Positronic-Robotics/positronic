from typing import Callable, Any

from ironic2 import SignalReader, SignalEmitter, Message


class MapSignalReader(SignalReader):

    def __init__(self, reader: SignalReader, func: Callable[[Any], Any]):
        self.reader = reader
        self.func = func

    def read(self):
        orig_message = self.reader.read()
        if orig_message is None:
            return None
        return Message(self.func(orig_message.data), orig_message.ts)


class MapSignalEmitter(SignalEmitter):

    def __init__(self, emitter: SignalEmitter, func: Callable[[Any], Any]):
        self.emitter = emitter
        self.func = func

    def emit(self, data: Any, ts: int | None = None) -> bool:
        return self.emitter.emit(self.func(data), ts)


def map(reader: SignalReader | SignalEmitter, func: Callable[[Any], Any]) -> SignalReader | SignalEmitter:
    if isinstance(reader, SignalReader):
        return MapSignalReader(reader, func)
    elif isinstance(reader, SignalEmitter):
        return MapSignalEmitter(reader, func)
    else:
        raise ValueError(f"Invalid reader type: {type(reader)}")


class ValueUpdated(SignalReader):
    """Wrapper around reader to signal whether the value we read is 'new'."""

    def __init__(self, reader: SignalReader):
        """By default, if original reader returns None, we return None."""
        self.reader = reader
        self.last_ts = None

    def read(self) -> Message | None:
        orig_message = self.reader.read()

        if orig_message is None:
            return None

        is_updated = orig_message.ts != self.last_ts
        self.last_ts = orig_message.ts

        return Message((orig_message.data, is_updated), self.last_ts)


class DefaultSignalReader(SignalReader):
    """Signal reader that returns a default value if no value is available."""

    def __init__(self, reader: SignalReader, default: Any, default_ts: int = 0):
        self.reader = reader
        self.default = default
        self.default_ts = default_ts

    def read(self) -> Message | None:
        msg = self.reader.read()
        if msg is None:
            return Message(self.default, self.default_ts)
        return msg
