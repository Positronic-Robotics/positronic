"""Fakes for the `policy` interfaces, for tests that drive a runtime on a virtual clock."""

from positronic.policy.base import Answer, NotAnswered


class DueAnswer(Answer):
    """An `Answer` that is done once the clock of `runtime` reaches `due_ns`."""

    def __init__(self, runtime, value, due_ns):
        self._runtime, self._value, self._due_ns = runtime, value, due_ns
        self._elapsed_sec = (due_ns - runtime.time_ns) / 1e9

    @property
    def elapsed_sec(self):
        if not self.done():
            raise NotAnswered()
        return self._elapsed_sec

    @property
    def execution_sec(self):
        return self.elapsed_sec

    def done(self):
        return self._runtime.time_ns >= self._due_ns

    def result(self):
        return self._value
