# S-expression Serialization

This package provides a generic, deterministic S-expression serializer for Python values and domain objects. It intentionally does not hard-code MetaMo types. MetaMo objects are supported because they are ordinary dataclasses and NumPy arrays.

## Output contract

The default API emits generic tagged S-expression data, not executable OpenCog Atomese. Atomese-shaped output is exposed by the separate `to_atomese`, `write_atomese`, and `HyperonAtomSpaceWriter` APIs and is not implied by ordinary dataclass tags such as `(MotivationalState ...)`.

## Public API

```python
from s_expression.serializer import to_sexpr, write_sexpr, parse_sexpr

value = {"z": [1, 2], "a": {"nested": True}}
text = to_sexpr(value)
write_sexpr(value, "output.sexpr")
parsed = parse_sexpr(text)
```

`to_ir()` provides the intermediate representation for decoupled introspection and formatting. `SExprRegistry` provides custom encoders without requiring changes to model classes.

## Supported values

- `None` → `nil`
- Booleans → `true` / `false`, configurable
- Integers and finite floats
- Strings with escaping for quotes, backslashes, control characters, and Unicode
- Lists and tuples
- Mappings with deterministic key sorting
- Dataclasses using declared field order
- Named tuples
- NumPy scalars and arrays, with optional dtype and shape metadata
- Custom domain types through `SExprRegistry` or `__sexpr__()`

Sets and unsupported objects are rejected because sets have no deterministic ordering and arbitrary Python objects would otherwise depend on memory addresses or `repr()` output.

## Non-finite values

NaN and infinity are rejected by default. `SExprConfig(nan_policy="symbol")` serializes them as `NaN`, `Infinity`, and `-Infinity`; `nan_policy="string"` serializes them as quoted strings.

## Mapping keys

Mappings may use supported scalar keys: `None`, booleans, integers, finite floats, strings, bytes, and NumPy scalars. Unsupported or mixed-domain keys are rejected or sorted by a documented stable type/value rule. Mapping entries are emitted under the configured `dict` tag in stable key order.

## Cycles

Direct and indirect recursive object graphs raise `CyclicReferenceError`. Repeated references that do not form a cycle are allowed.

## Current MetaMo core examples

The generic serializer discovers the fields on the actual core dataclasses.

```python
import numpy as np

from core.config import NUM_GOALS, NUM_MODULATORS
from core.state import MotivationalState, Action
from s_expression.serializer import to_sexpr

state = MotivationalState(
    G=np.full(NUM_GOALS, 0.5, dtype=float),
    M=np.full(NUM_MODULATORS, 0.5, dtype=float),
)

action = Action(
    id="explore",
    goal_correlations=np.full(NUM_GOALS, 0.1, dtype=float),
    delta_g=np.full(NUM_GOALS, 0.05, dtype=float),
    metadata={"generated_by": "example"},
)

print(to_sexpr(state))
print(to_sexpr(action))
print(to_sexpr([action]))
```

```text
;; state
(
  MotivationalState
  (G (Array (0.5 0.5)))
  (
    M
    (
      Array
      (
        0.5
        0.5
        0.5
        0.5
        0.5
        0.5
      )
    )
  )
  (
    schema
    (
      MotivationSchema
      (
        goals
        (
          MagusGoalSchema
          (primary_goals ())
          (anti_goals ())
          (individuation_name "individuation")
          (transcendence_name "transcendence")
        )
      )
      (
        modulators
        (
          ModulatorSchema
          (application_specific ())
          (
            openpsi_modulators
            (
              "valence"
              "arousal"
              "approach"
              "resolution"
              "threshold"
              "securing"
            )
          )
        )
      )
      (decision_profile_name "magus_additive_default")
      (appraisal_profile_name "openpsi_default")
    )
  )
)

;; action
(
  Action
  (id "explore")
  (goal_correlations (Array (0.1 0.1)))
  (delta_g (Array (0.05 0.05)))
  (
    schema
    (
      MotivationSchema
      (
        goals
        (
          MagusGoalSchema
          (primary_goals ())
          (anti_goals ())
          (individuation_name "individuation")
          (transcendence_name "transcendence")
        )
      )
      (
        modulators
        (
          ModulatorSchema
          (application_specific ())
          (
            openpsi_modulators
            (
              "valence"
              "arousal"
              "approach"
              "resolution"
              "threshold"
              "securing"
            )
          )
        )
      )
      (decision_profile_name "magus_additive_default")
      (appraisal_profile_name "openpsi_default")
    )
  )
  (metadata (dict ("generated_by" "example")))
)

;; candidates
(
  (
    Action
    (id "explore")
    (goal_correlations (Array (0.1 0.1)))
    (delta_g (Array (0.05 0.05)))
    (
      schema
      (
        MotivationSchema
        (
          goals
          (
            MagusGoalSchema
            (primary_goals ())
            (anti_goals ())
            (individuation_name "individuation")
            (transcendence_name "transcendence")
          )
        )
        (
          modulators
          (
            ModulatorSchema
            (application_specific ())
            (
              openpsi_modulators
              (
                "valence"
                "arousal"
                "approach"
                "resolution"
                "threshold"
                "securing"
              )
            )
          )
        )
        (decision_profile_name "magus_additive_default")
        (appraisal_profile_name "openpsi_default")
      )
    )
    (metadata (dict ("generated_by" "example")))
  )
)
```

The exact output depends on `SExprConfig`, especially dataclass field style, array tags, NumPy metadata, indentation, and compact formatting. The default output is deterministic and round-trip parser-compatible.

## Atomese compatibility

Default output from `to_sexpr` is generic tagged S-expression data, not executable OpenCog Atomese. It is deliberately kept separate from the Atomese APIs.

`to_atomese` and `write_atomese` produce Atomese-shaped text, while `HyperonAtomSpaceWriter` provides the separate Hyperon integration. This compatibility is validated against `hyperon 0.2.10`: generated text is parsed with `MeTTa.parse_all()`, and `HyperonAtomSpaceWriter` atoms are added through `metta.space().add_atom()`. Hyperon remains optional; generic S-expression serialization continues to work without it.

## Custom encoders

```python
from s_expression.serializer import SExprRegistry, to_sexpr

registry = SExprRegistry()

@registry.register(MyDomainType)
def encode_domain(value):
    return {"domain_id": value.identifier, "value": value.value}

print(to_sexpr(MyDomainType(...), registry=registry))
```

Custom encoders return values that the generic serializer understands. Registry encoders never require changing the source domain class.
