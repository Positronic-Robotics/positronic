"""Validated names and `Slugged[E]`, an `IntEnum` carried as its slug on the wire.

An unknown slug is REFUSED at the boundary (422) rather than landing on the `INVALID` sentinel, and
neither dump mode leaks the stored integer.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Annotated, Any, ClassVar, Self, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class Name(str):
    """A nonempty name without whitespace, preserving its concrete str subtype.

    A refusal reads `not <described_as>: <value>`. A subclass states how a person names its kind, with the article:
    `class EvalRef(Name, described_as='an eval name')`. A subclass that states none is named by its class.
    """

    __slots__ = ()
    _described_as: ClassVar[str]

    def __init_subclass__(cls, described_as: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._described_as = described_as if described_as is not None else f'a {cls.__name__}'

    def __new__(cls, value: str) -> Self:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not {cls._described_as}: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


E = TypeVar('E', bound=IntEnum)


def slug_of(member: IntEnum) -> str:
    return member.name.lower()


def members_by_slug(enum_cls: type[E]) -> dict[str, E]:
    """The wire vocabulary. `INVALID` is an in-memory sentinel and is deliberately absent."""
    return {m.name.lower(): m for m in enum_cls if m.value != 0}


class _SlugCodec:
    """The `Annotated` metadata behind `Slugged`; one instance serves every enum."""

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        enum_cls: type[IntEnum] = source_type
        by_slug = members_by_slug(enum_cls)

        def from_slug(value: str) -> IntEnum:
            return by_slug[value]

        def member_as_slug(value: object) -> object:
            if not isinstance(value, enum_cls):
                return value
            if value.value == 0:
                raise ValueError(f'{enum_cls.__name__}.{value.name} is an unset sentinel, not a wire value')
            return slug_of(value)

        # A literal schema over the slugs both rejects an unknown one and renders as a string enum in
        # the generated OpenAPI, which a plain validator function cannot do.
        parse_slug = core_schema.no_info_after_validator_function(from_slug, core_schema.literal_schema(list(by_slug)))
        return core_schema.json_or_python_schema(
            json_schema=parse_slug,
            # A member takes the slug check too, so a refused value gets one error, and it lists the slugs.
            python_schema=core_schema.no_info_before_validator_function(member_as_slug, parse_slug),
            serialization=core_schema.plain_serializer_function_ser_schema(
                slug_of, return_schema=core_schema.str_schema(), when_used='always'
            ),
        )


Slugged = Annotated[E, _SlugCodec()]
