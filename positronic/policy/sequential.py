"""Sequential composition of processors and codecs."""

from collections.abc import Generator
from typing import Any

from positronic.policy.base import SEQ, InputT, OutputT, Processor, ProcessorRun, Runtime
from positronic.policy.codec import Codec
from positronic.utils import flatten_dict


class Sequential(Processor[InputT, OutputT]):
    """Nest processors and codecs, with the first outermost.

    ``runtime.start(Sequential(A(...), B(...)), infer)`` passes ``infer`` to B's generator and B's
    generator to A. Each processor controls when and how often it sends inputs to its child.
    Codecs encode inputs and decode outputs at their position in the sequence. A codec around a
    generator preserves step timing; a codec around a callable runs wherever that callable runs.
    Nested sequences are flattened, so grouping does not change the dependencies passed to components.
    This sequence owns the generators it creates; external dependencies remain owned by their caller.
    """

    def __init__(self, first: Processor[InputT, OutputT] | Codec, /, *rest: Processor[Any, Any] | Codec) -> None:
        self._components = tuple(
            child
            for component in (first, *rest)
            for child in (component._components if type(component) is Sequential else (component,))
        )

    def run(self, runtime: Runtime, *dependencies: Any) -> ProcessorRun[InputT, OutputT]:
        children: list[ProcessorRun[Any, Any]] = []
        try:
            for component in reversed(self._components):
                if isinstance(component, Processor):
                    child = runtime.start(component, *dependencies)
                else:
                    child = component.wrap(*dependencies)
                if isinstance(child, Generator):
                    children.append(child)
                dependencies = (child,)
            call = dependencies[0]
            if isinstance(call, Generator):
                call = call.send
            value = yield
            while True:
                value = yield call(value)
        finally:
            for child in reversed(children):
                child.close()

    def meta(self) -> dict[str, Any]:
        """Combine component metadata; later components take precedence on shared keys."""
        meta = {}
        for component in self._components:
            values = component.meta() if isinstance(component, Processor) else component.meta
            meta.update(flatten_dict(values))
        return meta

    def to_spec(self) -> dict[str, Any]:
        return {SEQ: [component.to_spec() for component in self._components]}
