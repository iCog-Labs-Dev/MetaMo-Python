"""Generic, deterministic S-expression serialization for Python objects.

The base API emits generic S-expression data, not executable Atomese. Atomese-shaped
helpers are kept separate below and must be validated against the target AtomSpace API.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class SExprSerializationError(TypeError, ValueError):
    """Raised when a Python value cannot be serialized safely."""


class CyclicReferenceError(SExprSerializationError):
    """Raised when an object graph contains a cycle."""


class SExprParseError(ValueError):
    """Raised when source text is not valid S-expression syntax."""


@dataclass(frozen=True)
class SAtom:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("SAtom.value must be a string")
        if not _SYMBOL_RE.fullmatch(self.value):
            raise ValueError(f"Invalid unquoted atom: {self.value!r}")


@dataclass(frozen=True)
class SString:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("SString.value must be a string")


@dataclass(frozen=True)
class SList:
    items: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


SExprNode = SAtom | SString | SList
_SYMBOL_RE = re.compile(r"^(?:[A-Za-z_#+*/<>=!?$%&~^.#][A-Za-z0-9_+*/<>=!?$%&~^.#-]*|-[A-Za-z0-9_+*/<>=!?$%&~^.#-]+|-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$")


def _validate_symbol(value: str) -> None:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValueError(f"Invalid S-expression symbol: {value!r}")


@dataclass
class SExprConfig:
    """Controls formatting and supported serialization policies."""

    indent: int = 2
    compact: bool = False
    max_inline_items: int = 4
    nan_policy: str = "error"  # error, symbol, or string
    bool_style: str = "lower"  # lower, upper, or scheme
    array_tag: str | None = "Array"
    numpy_metadata: bool = False
    dict_tag: str = "dict"
    dict_entry_tag: str = "Entry"
    list_tag: str | None = None
    tuple_tag: str | None = None
    nil_symbol: str = "nil"
    dataclass_style: str = "tagged"  # tagged, list, or map
    dataclass_field_style: str = "tagged"  # tagged or positional
    dataclass_style_overrides: dict[str, str] | None = None
    include_repr_fields: bool = True
    float_precision: int | None = None
    ascii_only: bool = False
    sort_dict_keys: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.indent, bool) or not isinstance(self.indent, int) or self.indent < 0:
            raise ValueError("indent must be a non-negative integer")
        if not isinstance(self.compact, bool):
            raise ValueError("compact must be a boolean")
        if isinstance(self.max_inline_items, bool) or not isinstance(self.max_inline_items, int) or self.max_inline_items < 0:
            raise ValueError("max_inline_items must be a non-negative integer")
        if self.nan_policy not in {"error", "symbol", "string"}:
            raise ValueError("nan_policy must be 'error', 'symbol', or 'string'")
        if self.bool_style not in {"lower", "upper", "scheme"}:
            raise ValueError("bool_style must be 'lower', 'upper', or 'scheme'")
        if self.dataclass_style not in {"tagged", "list", "map"}:
            raise ValueError("dataclass_style must be 'tagged', 'list', or 'map'")
        if self.dataclass_field_style not in {"tagged", "positional"}:
            raise ValueError("dataclass_field_style must be 'tagged' or 'positional'")
        if self.float_precision is not None:
            if isinstance(self.float_precision, bool) or not isinstance(self.float_precision, int) or self.float_precision < 1:
                raise ValueError("float_precision must be a positive integer or None")
        for tag in (self.array_tag, self.list_tag, self.tuple_tag):
            if tag is not None:
                _validate_symbol(tag)
        for symbol in (self.dict_tag, self.dict_entry_tag, self.nil_symbol):
            _validate_symbol(symbol)
        if self.dataclass_style_overrides is None:
            self.dataclass_style_overrides = {}
        for class_name, style in self.dataclass_style_overrides.items():
            if not isinstance(class_name, str) or not class_name:
                raise ValueError("Dataclass override keys must be non-empty strings")
            if style not in {"tagged", "list", "map", "positional"}:
                raise ValueError(f"Invalid dataclass style for {class_name!r}")


Encoder = Callable[[Any], Any]
ContextEncoder = Callable[[Any, "Introspector"], Any]


class SExprRegistry:
    """Registry for custom domain encoders."""

    def __init__(self) -> None:
        self._encoders: dict[type, Encoder | ContextEncoder] = {}
        self._context_types: set[type] = set()

    def register(self, cls: type, encoder: Encoder | None = None) -> Any:
        if not isinstance(cls, type):
            raise TypeError("cls must be a type")

        def install(fn: Encoder) -> Encoder:
            if not callable(fn):
                raise TypeError("encoder must be callable")
            self._encoders[cls] = fn
            self._context_types.discard(cls)
            return fn

        return install if encoder is None else install(encoder)

    def register_with_context(self, cls: type, encoder: ContextEncoder | None = None) -> Any:
        if not isinstance(cls, type):
            raise TypeError("cls must be a type")

        def install(fn: ContextEncoder) -> ContextEncoder:
            if not callable(fn):
                raise TypeError("encoder must be callable")
            self._encoders[cls] = fn
            self._context_types.add(cls)
            return fn

        return install if encoder is None else install(encoder)

    def unregister(self, cls: type) -> None:
        self._encoders.pop(cls, None)
        self._context_types.discard(cls)

    def lookup(self, cls: type) -> tuple[Encoder | ContextEncoder, bool] | None:
        for candidate in cls.__mro__:
            if candidate in self._encoders:
                return self._encoders[candidate], candidate in self._context_types
        matches: list[tuple[int, Encoder | ContextEncoder, bool]] = []
        for registered, encoder in self._encoders.items():
            try:
                if issubclass(cls, registered):
                    matches.append((len(registered.__mro__), encoder, registered in self._context_types))
            except TypeError:
                continue
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        _, encoder, context = matches[0]
        return encoder, context

    def copy(self) -> "SExprRegistry":
        clone = SExprRegistry()
        clone._encoders = self._encoders.copy()
        clone._context_types = self._context_types.copy()
        return clone


DEFAULT_REGISTRY = SExprRegistry()


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and isinstance(getattr(type(value), "_fields", None), tuple)


def _stable_key(value: Any) -> tuple[Any, ...]:
    if value is None:
        return (0,)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int):
        return (2, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SExprSerializationError("Non-finite floats cannot be deterministic mapping keys")
        return (3, (0.0 if value == 0.0 else value).hex())
    if isinstance(value, str):
        return (4, value)
    if isinstance(value, bytes):
        return (5, value.hex())
    if np is not None and isinstance(value, np.generic):
        return _stable_key(value.item())
    raise SExprSerializationError(
        f"Cannot deterministically sort key type {type(value).__name__}; "
        "supported keys are None, booleans, numbers, strings, and bytes"
    )


def _float_text(value: float, config: SExprConfig) -> str:
    if not math.isfinite(value):
        if config.nan_policy == "error":
            raise SExprSerializationError("NaN and infinity disabled by nan_policy='error'")
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0.0:
        return "0.0"
    if config.float_precision is not None:
        return format(value, f".{config.float_precision}f")
    return repr(value)


def _escape_string(value: str, ascii_only: bool = False) -> str:
    out = ['"']
    for char in value:
        codepoint = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif char == "\b":
            out.append("\\b")
        elif char == "\f":
            out.append("\\f")
        elif codepoint < 0x20 or codepoint == 0x7F or 0xD800 <= codepoint <= 0xDFFF:
            out.append(f"\\u{codepoint:04x}")
        elif ascii_only and codepoint > 0x7E:
            if codepoint <= 0xFFFF:
                out.append(f"\\u{codepoint:04x}")
            else:
                adjusted = codepoint - 0x10000
                out.append(f"\\u{0xD800 + (adjusted >> 10):04x}\\u{0xDC00 + (adjusted & 0x3FF):04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


class Introspector:
    """Converts Python values into generic S-expression IR."""

    def __init__(self, config: SExprConfig | None = None, registry: SExprRegistry | None = None) -> None:
        self.config = config if config is not None else SExprConfig()
        self.registry = registry if registry is not None else DEFAULT_REGISTRY
        self._active_ids: set[int] = set()

    def convert(self, value: Any) -> SExprNode:
        if isinstance(value, (SAtom, SString, SList)):
            return value
        if value is None:
            return SAtom(self.config.nil_symbol)
        if isinstance(value, bool):
            if self.config.bool_style == "upper":
                return SAtom("TRUE" if value else "FALSE")
            if self.config.bool_style == "scheme":
                return SAtom("#t" if value else "#f")
            return SAtom("true" if value else "false")
        if isinstance(value, int):
            return SAtom(str(value))
        if isinstance(value, float):
            text = _float_text(value, self.config)
            return SString(text) if not math.isfinite(value) and self.config.nan_policy == "string" else SAtom(text)
        if isinstance(value, str):
            return SString(value)
        if isinstance(value, bytes):
            return SString(value.hex())
        if np is not None and isinstance(value, np.generic):
            return self.convert(value.item())

        custom = self.registry.lookup(type(value))
        if custom is not None:
            encoder, uses_context = custom
            return self._guarded(value, lambda: self.convert(
                encoder(value, self) if uses_context else encoder(value)
            ))
        if hasattr(value, "__sexpr__"):
            method = getattr(value, "__sexpr__")
            if not callable(method):
                raise SExprSerializationError("__sexpr__ must be callable")
            return self._guarded(value, lambda: self.convert(method()))
        if np is not None and isinstance(value, np.ndarray):
            return self._guarded(value, lambda: self._convert_ndarray(value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._guarded(value, lambda: self._convert_dataclass(value))
        if _is_namedtuple(value):
            return self._guarded(value, lambda: SList((SAtom(type(value).__name__), *(self.convert(x) for x in value))))
        if isinstance(value, Mapping):
            return self._guarded(value, lambda: self._convert_mapping(value))
        if isinstance(value, list):
            return self._guarded(value, lambda: self._convert_sequence(value, self.config.list_tag))
        if isinstance(value, tuple):
            return self._guarded(value, lambda: self._convert_sequence(value, self.config.tuple_tag))
        if isinstance(value, (set, frozenset)):
            raise SExprSerializationError("Sets are unsupported because their order is undefined")
        raise SExprSerializationError(f"Unsupported Python type: {_class_name(value)}")

    def _guarded(self, value: Any, callback: Callable[[], SExprNode]) -> SExprNode:
        identity = id(value)
        if identity in self._active_ids:
            raise CyclicReferenceError(f"Cyclic reference detected in {_class_name(value)}")
        self._active_ids.add(identity)
        try:
            return callback()
        finally:
            self._active_ids.remove(identity)

    def _convert_sequence(self, value: list[Any] | tuple[Any, ...], tag: str | None) -> SExprNode:
        elements = tuple(self.convert(item) for item in value)
        return SList(elements if tag is None else (SAtom(tag), *elements))

    def _convert_mapping(self, value: Mapping[Any, Any]) -> SExprNode:
        items = list(value.items())
        for key, _ in items:
            _stable_key(key)
        if self.config.sort_dict_keys:
            items.sort(key=lambda pair: _stable_key(pair[0]))
        entries = tuple(SList((self.convert(key), self.convert(item))) for key, item in items)
        return SList((SAtom(self.config.dict_tag), *entries))

    def _convert_dataclass(self, value: Any) -> SExprNode:
        class_name = type(value).__name__
        qualified_name = _class_name(value)
        style = self.config.dataclass_style_overrides.get(
            qualified_name,
            self.config.dataclass_style_overrides.get(class_name, self.config.dataclass_style),
        )
        fields = list(dataclasses.fields(value))
        if not self.config.include_repr_fields:
            fields = [field for field in fields if field.repr]
        pairs = [(field.name, self.convert(getattr(value, field.name))) for field in fields]

        if self.config.dataclass_field_style == "positional" or style in {"list", "positional"}:
            return SList((SAtom(class_name), *(node for _, node in pairs)))
        if style == "map":
            return SList((SAtom(class_name), *(SList((SAtom("Field"), SString(name), node)) for name, node in pairs)))
        return SList((SAtom(class_name), *(SList((SAtom(name), node)) for name, node in pairs)))

    def _convert_ndarray(self, value: Any) -> SExprNode:
        assert np is not None
        data = self.convert(value.tolist())
        if self.config.numpy_metadata:
            return SList((
                SAtom("ndarray"),
                SList((SAtom("dtype"), SString(str(value.dtype)))),
                SList((SAtom("shape"), SList(tuple(SAtom(str(size)) for size in value.shape)))),
                SList((SAtom("data"), data)),
            ))
        if self.config.array_tag is None:
            return data
        return SList((SAtom(self.config.array_tag), data))


class SExprFormatter:
    def __init__(self, config: SExprConfig | None = None) -> None:
        self.config = config if config is not None else SExprConfig()

    def format(self, node: SExprNode) -> str:
        return self._format(node, 0)

    def _format(self, node: SExprNode, depth: int) -> str:
        if isinstance(node, SAtom):
            return node.value
        if isinstance(node, SString):
            return _escape_string(node.value, self.config.ascii_only)
        if not isinstance(node, SList):
            raise SExprSerializationError(f"Invalid IR node: {type(node).__name__}")
        if not node.items:
            return "()"
        rendered = [self._format(item, depth + 1) for item in node.items]
        inline = "(" + " ".join(rendered) + ")"
        if self.config.compact or (len(node.items) <= self.config.max_inline_items and "\n" not in inline):
            return inline
        child = " " * (self.config.indent * (depth + 1))
        closing = " " * (self.config.indent * depth)
        return "(\n" + child + ("\n" + child).join(rendered) + "\n" + closing + ")"


def to_ir(value: Any, *, config: SExprConfig | None = None, registry: SExprRegistry | None = None) -> SExprNode:
    """Convert a Python value to generic S-expression IR."""
    return Introspector(config=config, registry=registry).convert(value)


def to_sexpr(value: Any, *, config: SExprConfig | None = None, registry: SExprRegistry | None = None) -> str:
    """Serialize a Python value as deterministic generic S-expression text."""
    actual = config if config is not None else SExprConfig()
    return SExprFormatter(actual).format(to_ir(value, config=actual, registry=registry))


def write_sexpr(value: Any, destination: str | Path | TextIO, *, config: SExprConfig | None = None,
                registry: SExprRegistry | None = None, encoding: str = "utf-8") -> None:
    """Write generic S-expression text to a file or text stream."""
    output = to_sexpr(value, config=config, registry=registry)
    if hasattr(destination, "write"):
        destination.write(output)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding=encoding)


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


def _combine_surrogates(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                result.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + low - 0xDC00))
                index += 2
                continue
        result.append(value[index])
        index += 1
    return "".join(result)


def _tokenize(source: str) -> list[_Token]:
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    tokens: list[_Token] = []
    index = 0
    while index < len(source):
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if character == ";":
            while index < len(source) and source[index] != "\n":
                index += 1
            continue
        if character in "()":
            tokens.append(_Token("LPAREN" if character == "(" else "RPAREN", character, index))
            index += 1
            continue
        if character == '"':
            start = index
            index += 1
            decoded: list[str] = []
            while index < len(source):
                character = source[index]
                if character == '"':
                    index += 1
                    tokens.append(_Token("STRING", _combine_surrogates("".join(decoded)), start))
                    break
                if character != "\\":
                    if ord(character) < 0x20:
                        raise SExprParseError(f"Unescaped control character at position {index}")
                    decoded.append(character)
                    index += 1
                    continue
                escape_position = index
                index += 1
                if index >= len(source):
                    raise SExprParseError(f"Unterminated escape at position {escape_position}")
                escape = source[index]
                index += 1
                simple = {'"': '"', "\\": "\\", "n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
                if escape in simple:
                    decoded.append(simple[escape])
                elif escape == "u":
                    digits = source[index:index + 4]
                    if len(digits) != 4 or any(digit not in "0123456789abcdefABCDEF" for digit in digits):
                        raise SExprParseError(f"Invalid Unicode escape at position {escape_position}")
                    decoded.append(chr(int(digits, 16)))
                    index += 4
                else:
                    raise SExprParseError(f"Unsupported escape '\\{escape}' at position {escape_position}")
            else:
                raise SExprParseError(f"Unterminated string at position {start}")
            continue
        start = index
        while index < len(source) and not source[index].isspace() and source[index] not in '();"':
            index += 1
        if index == start:
            raise SExprParseError(f"Unexpected character {source[index]!r} at position {index}")
        tokens.append(_Token("ATOM", source[start:index], start))
    return tokens


def parse_sexpr(source: str) -> SExprNode:
    """Parse exactly one generic S-expression into syntax IR; no type inference is performed."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    tokens = _tokenize(source)
    position = 0

    def parse_one() -> SExprNode:
        nonlocal position
        if position >= len(tokens):
            raise SExprParseError("Unexpected end of input")
        token = tokens[position]
        position += 1
        if token.kind == "STRING":
            return SString(token.value)
        if token.kind == "ATOM":
            return SAtom(token.value)
        if token.kind == "RPAREN":
            raise SExprParseError(f"Unexpected ')' at position {token.position}")
        if token.kind == "LPAREN":
            items: list[SExprNode] = []
            while True:
                if position >= len(tokens):
                    raise SExprParseError(f"Unclosed '(' at position {token.position}")
                if tokens[position].kind == "RPAREN":
                    position += 1
                    return SList(tuple(items))
                items.append(parse_one())
        raise SExprParseError(f"Unexpected token at position {token.position}")

    if not tokens:
        raise SExprParseError("Input is empty")
    result = parse_one()
    if position != len(tokens):
        raise SExprParseError(f"Unexpected extra expression at position {tokens[position].position}")
    return result


def _atomese_node(name: str, *children: SExprNode) -> SList:
    _validate_symbol(name)
    return SList((SAtom(name), *children))


def to_atomese_ir(value: Any, *, registry: SExprRegistry | None = None,
                  sort_dict_keys: bool = True) -> SExprNode:
    """Convert to proposed Atomese-shaped IR; validate constructors for the target environment."""
    actual_registry = registry if registry is not None else DEFAULT_REGISTRY
    active: set[int] = set()

    def guarded(obj: Any, callback: Callable[[], SExprNode]) -> SExprNode:
        identity = id(obj)
        if identity in active:
            raise CyclicReferenceError(f"Cyclic reference detected in {_class_name(obj)}")
        active.add(identity)
        try:
            return callback()
        finally:
            active.remove(identity)

    def convert(obj: Any) -> SExprNode:
        if isinstance(obj, (SAtom, SString, SList)):
            raise SExprSerializationError("Pass Python values, not generic IR, to Atomese conversion")
        if obj is None:
            return _atomese_node("ConceptNode", SString("None"))
        if isinstance(obj, bool):
            return _atomese_node("ConceptNode", SString("true" if obj else "false"))
        if isinstance(obj, int):
            return _atomese_node("NumberNode", SAtom(str(obj)))
        if isinstance(obj, float):
            if not math.isfinite(obj):
                raise SExprSerializationError("Atomese conversion rejects non-finite floats")
            return _atomese_node("NumberNode", SAtom(repr(obj)))
        if isinstance(obj, str):
            return _atomese_node("ConceptNode", SString(obj))
        if isinstance(obj, bytes):
            return _atomese_node("ConceptNode", SString(obj.hex()))
        if np is not None and isinstance(obj, np.generic):
            return convert(obj.item())
        custom = actual_registry.lookup(type(obj))
        if custom is not None:
            encoder, uses_context = custom
            if uses_context:
                raise SExprSerializationError("Context-aware encoders are available only in generic mode")
            return guarded(obj, lambda: convert(encoder(obj)))
        if hasattr(obj, "__sexpr__"):
            method = getattr(obj, "__sexpr__")
            if not callable(method):
                raise SExprSerializationError("__sexpr__ must be callable")
            return guarded(obj, lambda: convert(method()))
        if np is not None and isinstance(obj, np.ndarray):
            return guarded(obj, lambda: _atomese_node("ListLink", *(convert(value) for value in obj.tolist())))
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            def dataclass_node() -> SExprNode:
                fields = [
                    _atomese_node("ListLink", _atomese_node("ConceptNode", SString(field.name)), convert(getattr(obj, field.name)))
                    for field in dataclasses.fields(obj) if field.repr
                ]
                return _atomese_node("ListLink", _atomese_node("ConceptNode", SString(type(obj).__name__)), *fields)
            return guarded(obj, dataclass_node)
        if _is_namedtuple(obj):
            return guarded(obj, lambda: _atomese_node(
                "ListLink", _atomese_node("ConceptNode", SString(type(obj).__name__)), *(convert(value) for value in obj)
            ))
        if isinstance(obj, Mapping):
            def mapping_node() -> SExprNode:
                items = list(obj.items())
                if sort_dict_keys:
                    items.sort(key=lambda pair: _stable_key(pair[0]))
                return _atomese_node("ListLink", *(
                    _atomese_node("ListLink", convert(key), convert(item)) for key, item in items
                ))
            return guarded(obj, mapping_node)
        if isinstance(obj, (list, tuple)):
            return guarded(obj, lambda: _atomese_node("ListLink", *(convert(value) for value in obj)))
        if isinstance(obj, (set, frozenset)):
            raise SExprSerializationError("Sets are unsupported in deterministic Atomese conversion")
        raise SExprSerializationError(f"Unsupported Atomese value type: {_class_name(obj)}")

    return convert(value)


def to_atomese(value: Any, *, registry: SExprRegistry | None = None, indent: int = 2,
               sort_dict_keys: bool = True) -> str:
    return SExprFormatter(SExprConfig(indent=indent)).format(
        to_atomese_ir(value, registry=registry, sort_dict_keys=sort_dict_keys)
    )


def write_atomese(value: Any, destination: str | Path | TextIO, *, registry: SExprRegistry | None = None,
                  encoding: str = "utf-8", sort_dict_keys: bool = True) -> None:
    output = to_atomese(value, registry=registry, sort_dict_keys=sort_dict_keys)
    if hasattr(destination, "write"):
        destination.write(output)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding=encoding)


__all__ = [
    "SAtom", "SString", "SList", "SExprNode", "SExprConfig", "SExprRegistry",
    "DEFAULT_REGISTRY", "SExprSerializationError", "CyclicReferenceError",
    "SExprParseError", "Introspector", "SExprFormatter", "to_ir", "to_sexpr",
    "write_sexpr", "parse_sexpr", "to_atomese_ir", "to_atomese", "write_atomese",
    "HyperonAtomSpaceWriter",
]


class HyperonAtomSpaceWriter:
    """Optional Hyperon adapter; validate API compatibility with the installed version."""

    def __init__(self, metta: Any | None = None) -> None:
        try:
            from hyperon import MeTTa  # type: ignore[import-not-found]
            from hyperon.atoms import E, S, ValueAtom  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Hyperon is required for HyperonAtomSpaceWriter") from exc
        self._E, self._S, self._ValueAtom = E, S, ValueAtom
        self.metta = metta if metta is not None else MeTTa()

    def ir_to_hyperon_atom(self, node: SExprNode) -> Any:
        if isinstance(node, SString):
            return self._ValueAtom(node.value)
        if isinstance(node, SAtom):
            return self._S(node.value)
        if isinstance(node, SList):
            return self._E(*(self.ir_to_hyperon_atom(item) for item in node.items))
        raise SExprSerializationError(f"Unsupported IR node: {type(node).__name__}")

    def add_ir(self, node: SExprNode) -> Any:
        atom = self.ir_to_hyperon_atom(node)
        self.metta.space().add_atom(atom)
        return atom

    def add_sexpr(self, source: str) -> Any:
        return self.add_ir(parse_sexpr(source))

    def add_python_value(self, value: Any, *, config: SExprConfig | None = None,
                         registry: SExprRegistry | None = None) -> Any:
        return self.add_ir(to_ir(value, config=config, registry=registry))
