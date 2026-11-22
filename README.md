# MetaMo-Python

MetaMo-Python includes a generic, deterministic S-expression serializer in `s_expression.serializer`.

## Generic S-expression API

```python
from s_expression.serializer import (
    SExprConfig,
    parse_sexpr,
    to_sexpr,
    write_sexpr,
)

text = to_sexpr({"b": [1, 2], "a": True})
write_sexpr([1, 2, 3], "values.sexpr")
parsed = parse_sexpr(text)

compact = SExprConfig(compact=True, numpy_metadata=True)
```

The default output is generic tagged S-expression data, not executable OpenCog Atomese. Dataclasses are serialized by field name and declared field order; lists and tuples preserve order; mappings use deterministic key sorting; NumPy arrays are supported with optional dtype and shape metadata.

## Atomese and Hyperon compatibility

The separate `to_atomese()`, `write_atomese()`, `to_atomese_ir()`, and `HyperonAtomSpaceWriter` APIs emit and add Atomese-shaped expressions for the supported Python values. These APIs use standard `Node`, `NumberNode`, `ConceptNode`, and `ListLink` constructor forms and were verified against **Hyperon 0.2.10 on Python 3.11**.

The runtime test confirms that:

- `to_atomese()` text can be parsed by `hyperon.MeTTa.parse_all()`.
- `HyperonAtomSpaceWriter` can construct Hyperon atoms for generic IR.
- `HyperonAtomSpaceWriter.add_python_value()` stores atoms in the Hyperon grounding space.

Install Hyperon only in a Python 3.11-compatible environment. Without it, generic S-expression APIs remain available and the runtime integration test is skipped by pytest.

Additional guidance is available in `s_expression/README.md`.

## Testing

```shell
python -m pytest tests/test_serializer_acceptance.py tests/test_golden_sexpr_examples.py -q
```
