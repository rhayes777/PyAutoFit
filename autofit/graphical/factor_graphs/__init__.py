from typing import Union

from .abstract import AbstractNode, Variable, Plate
from .factor import \
    FactorValue, AbstractFactor, Factor, DeterministicFactor
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