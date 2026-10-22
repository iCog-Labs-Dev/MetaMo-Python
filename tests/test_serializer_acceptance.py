"""Acceptance tests for the generic deterministic S-expression serializer."""

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pytest

from core.serializer import (
    CyclicReferenceError,
    SExprConfig,
    SExprSerializationError,
    SList,
    SString,
    parse_sexpr,
    to_ir,
    to_sexpr,
    write_sexpr,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "nil"),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (42, "42"),
        (-7, "-7"),
        (1.5, "1.5"),
        (-0.0, "0.0"),
        ("hello", '"hello"'),
        ("", '""'),
    ],
)
def test_supported_primitives(value, expected):
    assert to_sexpr(value) == expected


def test_non_finite_floats_rejected_by_default():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(SExprSerializationError):
            to_sexpr(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        'contains "quotes"',
        "backslash \\",
        "line one\nline two",
        "carriage\rreturn",
        "tab\tcharacter",
        "backspace\bformfeed\f",
        "control:\x01",
        "Unicode: አማርኛ café 😀",
    ],
)
def test_escaped_strings_are_deterministic_and_parseable(value):
    result = to_sexpr(value)

    assert result == to_sexpr(value)
    assert isinstance(parse_sexpr(result), SString)


def test_ascii_only_string_configuration():
    config = SExprConfig(ascii_only=True)
    result = to_sexpr("café 😀", config=config)

    assert result.isascii()
    assert isinstance(parse_sexpr(result), SString)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([], "()"),
        ((), "()"),
        ({}, "(dict)"),
        ([1, 2, 3], "(1 2 3)"),
        ((1, "x"), '(1 "x")'),
        ({"b": 2, "a": 1}, '(dict ("a" 1) ("b" 2))'),
    ],
)
def test_container_golden_outputs(value, expected):
    assert to_sexpr(value) == expected


def test_mapping_order_does_not_depend_on_insertion_order():
    first = {"z": 3, "a": 1, "m": 2}
    second = {"m": 2, "z": 3, "a": 1}

    assert to_sexpr(first) == to_sexpr(second)


def test_nested_containers():
    value = [1, {"items": [2, 3]}, (4, 5)]

    output = to_sexpr(value)

    assert isinstance(parse_sexpr(output), SList)


@dataclass
class Inner:
    name: str
    count: int


@dataclass
class Outer:
    inner: Inner
    enabled: bool


def test_nested_dataclasses():
    value = Outer(Inner("example", 3), True)

    output = to_sexpr(value)

    assert output == (
        '(Outer (inner (Inner (name "example") (count 3))) '
        '(enabled true))'
    )
    assert isinstance(parse_sexpr(output), SList)


def test_positional_dataclass_configuration():
    config = SExprConfig(dataclass_field_style="positional")

    assert to_sexpr(Inner("example", 3), config=config) == (
        '(Inner "example" 3)'
    )


def test_numpy_array_default_representation():
    value = np.array([1, 2, 3], dtype=np.int32)

    output = to_sexpr(value)

    assert isinstance(parse_sexpr(output), SList)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (np.int32, (3,)),
        (np.float64, (2, 2)),
        (np.bool_, (2,)),
    ],
)
def test_numpy_dtype_and_shape_metadata(dtype, shape):
    value = np.zeros(shape, dtype=dtype)
    config = SExprConfig(numpy_metadata=True)

    output = to_sexpr(value, config=config)

    assert "(dtype " in output
    assert "(shape " in output
    assert "(data " in output
    assert np.dtype(dtype).name in output
    assert isinstance(parse_sexpr(output), SList)


def test_numpy_empty_array_with_metadata():
    value = np.array([], dtype=np.float32)
    config = SExprConfig(numpy_metadata=True)

    output = to_sexpr(value, config=config)

    assert "float32" in output
    assert "(shape (0))" in output
    assert isinstance(parse_sexpr(output), SList)


def test_unsupported_object_is_rejected():
    class Unsupported:
        pass

    with pytest.raises(SExprSerializationError):
        to_sexpr(Unsupported())


def test_unsupported_mapping_key_is_rejected():
    with pytest.raises(SExprSerializationError):
        to_sexpr({("tuple",): "value"})


def test_direct_list_cycle_is_rejected():
    value = []
    value.append(value)

    with pytest.raises(CyclicReferenceError):
        to_sexpr(value)


def test_indirect_container_cycle_is_rejected():
    first = []
    second = {"back": first}
    first.append(second)

    with pytest.raises(CyclicReferenceError):
        to_sexpr(first)


def test_repeated_non_cyclic_reference_is_allowed():
    shared = [1, 2]
    value = [shared, shared]

    output = to_sexpr(value)

    assert output == to_sexpr(value)
    assert isinstance(parse_sexpr(output), SList)


def assert_balanced_parentheses(text):
    depth = 0
    in_string = False
    escaped = False

    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            assert depth >= 0, "Closing parenthesis without matching opener"

    assert not in_string, "Unterminated string literal"
    assert not escaped, "Incomplete escape sequence"
    assert depth == 0, "Unbalanced parentheses"


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        123,
        3.25,
        'quoted "text"\n',
        [],
        [1, [2, 3]],
        {"z": [1], "a": {"nested": True}},
        np.array([[1, 2], [3, 4]], dtype=np.int16),
        Outer(Inner("x", 2), False),
    ],
)
def test_repeated_serialization_is_byte_identical(value):
    outputs = [to_sexpr(value) for _ in range(100)]

    assert all(output == outputs[0] for output in outputs)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "text (with parentheses)",
        [1, [2, 3]],
        {"key": 'value "quoted"'},
        np.array([[1, 2], [3, 4]]),
        Outer(Inner("nested", 1), True),
    ],
)
def test_output_is_balanced_and_parser_accepts_it(value):
    output = to_sexpr(value)

    assert_balanced_parentheses(output)
    parse_sexpr(output)


def test_write_sexpr_matches_to_sexpr():
    value = {"b": [1, 2], "a": "hello"}
    stream = StringIO()

    write_sexpr(value, stream)

    assert stream.getvalue() == to_sexpr(value)


def test_to_ir_returns_a_node():
    result = to_ir([1, 2])

    assert isinstance(result, SList)
