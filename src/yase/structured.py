"""Dependency-free structured-output contracts for vision-language stages."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class StructuredQuery:
    """A VLM request whose answer must conform to a small JSON schema."""

    prompt: str
    schema: Mapping[str, Any]
    max_new_tokens: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("structured query prompt must not be empty")
        if not isinstance(self.schema, Mapping) or not self.schema:
            raise ValueError("structured query schema must be a non-empty mapping")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1 when provided")
        object.__setattr__(self, "schema", dict(self.schema))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "schema": dict(self.schema),
            "max_new_tokens": self.max_new_tokens,
            "metadata": dict(self.metadata),
        }


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected is not None:
        expected_types = (expected,) if isinstance(expected, str) else tuple(expected)
        if not any(_type_matches(value, item) for item in expected_types):
            raise ValueError(f"structured output {path} must have type {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"structured output {path} is not an allowed value")
    if isinstance(value, Mapping):
        required = schema.get("required", ())
        for key in required:
            if key not in value:
                raise ValueError(f"structured output {path}.{key} is required")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"structured schema {path} properties must be a mapping")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ValueError(
                    f"structured output {path} has unknown fields {sorted(unknown)}"
                )
        for key, child_schema in properties.items():
            if key in value:
                if not isinstance(child_schema, Mapping):
                    raise ValueError(
                        f"structured schema {path}.{key} must be a mapping"
                    )
                _validate(value[key], child_schema, f"{path}.{key}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        item_schema = schema.get("items")
        if item_schema is not None:
            if not isinstance(item_schema, Mapping):
                raise ValueError(f"structured schema {path}.items must be a mapping")
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]")


def parse_structured_output(text: str, schema: Mapping[str, Any]) -> Any:
    """Parse JSON or fenced JSON and validate it against a small JSON schema."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("structured model output must be non-empty text")
    candidate = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for marker in ("{", "["):
            position = candidate.find(marker)
            if position < 0:
                continue
            try:
                value, _ = decoder.raw_decode(candidate[position:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("structured model output is not valid JSON") from None
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("structured schema must be a non-empty mapping")
    _validate(value, schema, "$")
    return value


__all__ = ["StructuredQuery", "parse_structured_output"]
