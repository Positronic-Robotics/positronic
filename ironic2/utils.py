import time
from typing import Callable, Any, Dict, Literal, Sequence, Tuple

from ironic2 import NoValueType, SignalReader, SignalEmitter, NoValue, Message


class Printer:
    def __init__(self, **pipes: SignalReader):
        self.pipes = pipes

    def run(self, should_stop: SignalReader, sleep_time: float = 0.01):
        while not should_stop.value().data:
            for pipe_name, pipe in self.pipes.items():
                print(f"{pipe_name}: {pipe.value()}")
            time.sleep(sleep_time)


class LambdaSignalReader(SignalReader):
    def __init__(self, reader: SignalReader, func: Callable[[Any], Any]):
        self.reader = reader
        self.func = func

    def value(self):
        orig_message = self.reader.value()
        if orig_message is NoValue:
            return NoValue
        return Message(self.func(orig_message.data), orig_message.ts)


class LambdaSignalEmitter(SignalEmitter):
    def __init__(self, emitter: SignalEmitter, func: Callable[[Any], Any]):
        self.emitter = emitter
        self.func = func

    def emit(self, message: Message):
        self.emitter.emit(Message(self.func(message.data), message.ts))


def map(reader: SignalReader | SignalEmitter, func: Callable[[Any], Any]) -> SignalReader | SignalEmitter:
    if isinstance(reader, SignalReader):
        return LambdaSignalReader(reader, func)
    elif isinstance(reader, SignalEmitter):
        return LambdaSignalEmitter(reader, func)
    else:
        raise ValueError(f"Invalid reader type: {type(reader)}")


class ValueUpdated(SignalReader):
    def __init__(self, reader: SignalReader):
        self.reader = reader
        self.last_ts = 0

    def value(self) -> Tuple[Message | NoValueType, bool]:
        orig_message = self.reader.value()

        if orig_message is NoValue:
            return NoValue, False  # TODO: should we return False or True?

        is_updated = orig_message.ts != self.last_ts
        self.last_ts = orig_message.ts

        return orig_message, is_updated