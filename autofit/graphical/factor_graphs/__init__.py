from typing import Union
from .factor import AbstractFactor, Factor, DeterministicFactor
from .abstract import (
    Value,
    FactorValue,
    AbstractNode,
    Variable,
    variables, 
    VariableData,
    Plate,
)

from .graph import FactorGraph
from .transform import (
    VariableTransform,
    FullCholeskyTransform,
    identity_transform,
    TransformedNode,
)
from .jacobians import FactorJac, FactorJacobian, DeterministicFactorJacobian

# FactorNode = Union[Factor, FactorJacobian]
# DeterministicFactorNode = Union[DeterministicFactor, DeterministicFactorJacobian]
