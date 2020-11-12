from typing import Union

from .abstract import \
    FactorValue, JacobianValue, HessianValue, AbstractNode, Variable, Plate
from .factor import \
    AbstractFactor, Factor, DeterministicFactor
from .jacobians import \
    FactorJacobian, DeterministicFactorJacobian
from .graph import FactorGraph

FactorNode = Union[
    Factor, 
    FactorJacobian
]
DeterministicFactorNode = Union[
    DeterministicFactor,
    DeterministicFactorJacobian
    ]