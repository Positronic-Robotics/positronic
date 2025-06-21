from typing import Callable, Any, Tuple

from ironic2 import SignalReader, SignalEmitter, Message


class MapSignalReader(SignalReader):

    def __init__(self, reader: SignalReader, func: Callable[[Any], Any]):
        self.reader = reader
        self.func = func

    def value(self):
        orig_message = self.reader.value()
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


_NO_DEFAULT_SENTINEL = object()


class ValueUpdated(SignalReader):
    """Wrapper around reader to signal whether the value we read is 'new'."""

    def __init__(self, reader: SignalReader, default_value=_NO_DEFAULT_SENTINEL):
        """By default, if original reader returns None, we return None.
        If default_value is overriden, we will return (default_value, False) Message instead."""
        self.reader = reader
        self.last_ts = 0
        self._default_value = default_value

    def value(self) -> Tuple[Message | None, bool]:
        orig_message = self.reader.value()

        if orig_message is None:
            if self._default_value == _NO_DEFAULT_SENTINEL:
                return None
            else:
                return self._default_value, False

        is_updated = orig_message.ts != self.last_ts
        self.last_ts = orig_message.ts

        return Message((orig_message, is_updated), self.last_ts)
