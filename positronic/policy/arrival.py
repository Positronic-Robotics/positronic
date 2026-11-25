"""Move a declared image transform into the temporal stack that feeds it."""

import functools
import operator

from positronic.keys import IMAGE_PREFIX
from positronic.policy.base import FieldUse, Processor
from positronic.policy.codecs import Codec, TransformImages, TransformOnArrival
from positronic.policy.processors import TemporalStack
from positronic.policy.sequential import Sequential

Component = Processor | Codec


def transform_frames_on_arrival(stack: Processor) -> Processor:
    """``stack`` with each camera frame of a temporal stack transformed as it arrives, when the result is the same.

    A camera field moves only when every component between the stack and the transform forwards it
    (``FieldUse.FORWARDED``).
    """
    if not isinstance(stack, Sequential):
        return stack
    components = stack.components
    result: list[Component] = []
    for index, component in enumerate(components):
        if isinstance(component, TemporalStack) and (arrivals := _arrivals(component, components[index + 1 :])):
            component = component.with_arrival(functools.reduce(operator.or_, arrivals).encode)
        result.append(component)
    return stack if all(new is old for new, old in zip(result, components, strict=True)) else Sequential(*result)


def _arrivals(stack: TemporalStack, inner: tuple[Component, ...]) -> list[Codec]:
    """The transforms of ``stack``'s camera fields that ``inner`` lets ``stack`` apply as each frame arrives.

    A field stops at the first transform that selects it: a second one sees what the first made.
    """
    fields = [field for field in stack.keys if field.startswith(IMAGE_PREFIX)]
    arrivals: list[Codec] = []
    for component in inner:
        if isinstance(component, TransformImages):
            selected = [field for field in fields if component.selects(field)]
            if selected:
                arrivals.append(TransformOnArrival(component, selected))
            fields = [field for field in fields if field not in selected]
        else:
            fields = [field for field in fields if component.field_use(field) is FieldUse.FORWARDED]
        if not fields:
            break
    return arrivals
