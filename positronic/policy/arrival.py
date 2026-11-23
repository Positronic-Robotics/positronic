"""Move a declared image transform in front of the temporal stack that feeds it."""

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
    moved: set[tuple[int, str]] = set()
    result: list[Component] = []
    for index, component in enumerate(components):
        if isinstance(component, TemporalStack):
            result.extend(_arrivals(component, components[index + 1 :], moved))
        result.append(component)
    return stack if len(result) == len(components) else Sequential(*result)


def _arrivals(stack: TemporalStack, inner: tuple[Component, ...], moved: set[tuple[int, str]]) -> list[Codec]:
    """The transforms of ``stack``'s camera fields that ``inner`` lets run in front of ``stack``.

    A later temporal stack stacks a transformed frame as it stacks the original one, so the walk goes past it. A
    field stops at the first transform that selects it: a second one sees what the first made.
    """
    fields = [field for field in stack.keys if field.startswith(IMAGE_PREFIX)]
    arrivals: list[Codec] = []
    for component in inner:
        if isinstance(component, TransformImages):
            selected = [field for field in fields if component.selects(field)]
            new = [field for field in selected if (id(component), field) not in moved]
            if new:
                moved.update((id(component), field) for field in new)
                arrivals.append(TransformOnArrival(component, new))
            fields = [field for field in fields if field not in selected]
        elif not isinstance(component, TemporalStack):
            fields = [field for field in fields if component.field_use(field) is FieldUse.FORWARDED]
        if not fields:
            break
    return arrivals
