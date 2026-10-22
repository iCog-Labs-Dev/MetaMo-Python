"""Compatibility wrapper for the generic S-expression serializer.

The canonical implementation lives in the s_expression package so the project can
provide a dedicated serializer namespace without hard-coding MetaMo types.
"""

from s_expression.serializer import *  

__all__ = [
    "SAtom",
    "SString",
    "SList",
    "SExprNode",
    "SExprConfig",
    "SExprRegistry",
    "DEFAULT_REGISTRY",
    "SExprSerializationError",
    "CyclicReferenceError",
    "SExprParseError",
    "Introspector",
    "SExprFormatter",
    "to_ir",
    "to_sexpr",
    "write_sexpr",
    "parse_sexpr",
    "to_atomese_ir",
    "to_atomese",
    "write_atomese",
    "HyperonAtomSpaceWriter",
]
