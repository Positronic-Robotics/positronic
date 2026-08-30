"""Test doubles for the policy API: an answer that is already answered, and a runtime that answers inline."""

from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack
from functools import partial
from typing import Any

import pytest

from positronic.policy.base import Answer, Fn, Policy, Runtime, Session


class Done(Answer):
    """The answer to a call whose work ran inside it."""

    def __init__(self, value: Any):
        self._value = value

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._value


class InlineRuntime(Runtime):
    """Runs each call on the calling thread, so the answer it gives is already answered."""

    def __init__(self, functions: Mapping[str, Callable[..., Any]]):
        self._fns: Mapping[str, Fn] = {name: partial(self._call, fn) for name, fn in functions.items()}

    @staticmethod
    def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Answer:
        return Done(fn(*args, **kwargs))

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns


@pytest.fixture
def open_session() -> Generator[Callable[[Policy], Session], None, None]:
    """Opens a policy's session over an inline runtime; every one it opened is closed at teardown."""
    closing = ExitStack()

    def make(policy: Policy) -> Session:
        session = policy.new_session(InlineRuntime(closing.enter_context(policy.episode())))
        closing.callback(session.close)
        return session

    yield make
    closing.close()
