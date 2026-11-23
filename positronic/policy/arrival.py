"""Move a declared image transform in front of the temporal stack that feeds it."""

from positronic.keys import IMAGE_PREFIX
from positronic.policy.base import FieldUse, Processor
from positronic.policy.codecs import Codec, TransformImages, TransformOnArrival
from positronic.policy.processors import TemporalStack
from positronic.policy.sequential import Sequential

Component = Processor | Codec


def transform_frames_on_arrival(stack: Processor) -> Processor:
    """``stack`` with each camera frame of a temporal stack transformed as it arrives, when the result is the same.

    A server can declare an image transform inside a temporal stack. The transform then runs on the whole history at
    each call. This function puts a ``TransformOnArrival`` in front of the stack for each camera field that reaches
    the transform unchanged, so the main loop transforms each frame one time and the call transforms none. The
    transform gives each frame of a stack the same result as the frame alone, and gives a transformed frame back
    unchanged. So the policy gets the same observation. A component between the two that does not promise to forward
    the field (``FieldUse.FORWARDED``) keeps the field out.
    """
    if not isinstance(stack, Sequential):
        return stack
    components = stack.components
    moved: set[tuple[int, str]] = set()
    result: list[Component] = []
    for index, component in enumerate(components):
        if isinstance(component, TemporalStack):
            arrival = _arrival(component, components[index + 1 :], moved)
            if arrival is not None:
                result.append(arrival)
        result.append(component)
    return stack if len(result) == len(components) else Sequential(*result)


def _arrival(stack: TemporalStack, inner: tuple[Component, ...], moved: set[tuple[int, str]]) -> Codec | None:
    """The transform of ``stack``'s camera fields that ``inner`` lets run in front of ``stack``, if any.

    A later temporal stack stacks a transformed frame as it stacks the original one, so the walk goes past it.
    """
    fields = [field for field in stack.keys if field.startswith(IMAGE_PREFIX)]
    for component in inner:
        if isinstance(component, TransformImages):
            selected = [field for field in fields if component.selects(field) and (id(component), field) not in moved]
            # The walk ends at the first transform: a second one sees what the first made.
            if not selected:
                return None
            moved.update((id(component), field) for field in selected)
            return TransformOnArrival(component, selected)
        if not isinstance(component, TemporalStack):
            fields = [field for field in fields if component.field_use(field) is FieldUse.FORWARDED]
        if not fields:
            return None
    return None
