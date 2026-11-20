# S-expression Serialization

This package provides a generic, deterministic S-expression serializer for Python values and domain objects. It is intentionally generic and does not hard-code MetaMo classes. MetaMo models are supported because they are ordinary dataclasses and NumPy arrays, not because they are special-cased.

## Output contract

The default API emits generic S-expressions, not executable OpenCog Atomese. For Atomese-shaped output, use `to_atomese` and `write_atomese`, which are intentionally separate helpers.

## Public API

```python
from s_expression.serializer import to_sexpr, write_sexpr, parse_sexpr

value = {"z": [1, 2], "a": {"nested": True}}
text = to_sexpr(value)
write_sexpr(value, "output.sexpr")
parsed = parse_sexpr(text)
```

## Supported values

- `None` -> `nil`
- booleans -> `true` / `false`
- ints and finite floats
- strings with escaping for quotes, backslashes, control characters, and Unicode
- lists and tuples
- mappings with stable deterministic key ordering
- dataclass instances using declared field order
- NumPy scalars and arrays, with optional metadata
- custom domain types via `SExprRegistry` or `__sexpr__`

## Non-finite float policy

NaN and infinity are rejected by default with `SExprSerializationError`. This keeps serialized values safe and deterministic.

## Example: MetaMo state

```python
import numpy as np
from core.config import NUM_MODULATORS
from core.state import MotivationalState, Stimulus, Action
from s_expression.serializer import to_sexpr

state = MotivationalState(
    G=np.array([0.5, 0.5, 0.8, 0.6, 0.4, 0.3, 0.9, 0.2]),
    M=np.full(NUM_MODULATORS, 0.5),
)

stimulus = Stimulus(novelty=0.8, conduciveness=0.5, risk=0.2, effort=0.3)

action = Action(
    id="explore",
    goal_correlations=np.array([0.1, 0.2]),
    risk_estimate=0.6,
    delta_g=np.array([0.05, -0.05]),
)

print(to_sexpr(state))
print(to_sexpr(stimulus))
print(to_sexpr(action))
```

## Golden outputs

The generated examples preserve stable ordering, balanced parentheses, and parser compatibility. See the module `s_expression/golden_examples.py` for example values and `tests/test_golden_sexpr_examples.py` for verification.
