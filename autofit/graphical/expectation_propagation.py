
from abc import ABC, abstractmethod
from typing import (
    Dict, Tuple, Optional, NamedTuple, Iterator, List,
    Callable
)

import numpy as np

from autofit.graphical.messages.abstract import AbstractMessage

from autofit.graphical.factor_graphs import (
    Factor, AbstractNode, FactorGraph)
from autofit.mapper.variable import Variable
from autofit.graphical.utils import Status
from autofit.graphical.mean_field import MeanField, FactorApproximation 


class EPMeanField(AbstractNode):
    '''
    '''
    def __init__(
            self,
            factor_graph: FactorGraph,
            factor_mean_field: Dict[Factor, MeanField]
    ):
        self._factor_graph = factor_graph
        self._factor_mean_field = factor_mean_field
        variable_factor = {}
        for factor, vs in factor_graph.factor_all_variables.items():
            for v in vs:
                variable_factor.setdefault(v, set()).add(factor)
        self._variable_factor = variable_factor
        
        super().__init__(**self.factor_graph._kwargs)

    @property 
    def name(self):
        return f"EP_{self.factor_graph.name}"

    @property
    def variables(self): 
        return self.factor_graph.variables

    @property
    def deterministic_variables(self): 
        return self.factor_graph.deterministic_variables

    @property
    def variable_names(self) -> Dict[str, Variable]: 
        return self.factor_graph.variable_names
 
    @property
    def factor_mean_field(self) -> Dict[Factor, MeanField]:
        return self._factor_mean_field.copy()

    @property
    def factor_graph(self) -> FactorGraph:
        return self._factor_graph

    @classmethod
    def from_approx_dists(
            cls,
            factor_graph: FactorGraph,
            approx_dists: Dict[Variable, AbstractMessage],
    ) -> "EPMeanField":
        factor_mean_field = {
            factor: MeanField({
                v: approx_dists[v] for v in factor.variables})
            for factor in factor_graph.factors}

        return cls(
            factor_graph,
            factor_mean_field)

    from_kws = from_approx_dists
    
    def factor_approximation(self, factor: Factor) -> FactorApproximation:
        factor_mean_field = self.factor_mean_field
        factor_dist = factor_mean_field.pop(factor)
        cavity_dist = MeanField.prod(
            {v: 1. for v in factor_dist.all_variables},
            *(dist for fac, dist in factor_mean_field.items()))
        # cavity_dist.log_norm = 0.
        model_dist = factor_dist.prod(cavity_dist)

        return FactorApproximation(
            factor, cavity_dist, factor_dist, model_dist)

    def project_factor_approx(
        self, projection: FactorApproximation, status: Optional[Status] = None,
    ) -> "EPMeanField":
        """
        """
        factor_mean_field = self.factor_mean_field
        factor_mean_field[projection.factor] = projection.factor_dist

        new_approx = type(self)(
            factor_graph=self._factor_graph,
            factor_mean_field=factor_mean_field)
        return new_approx, status

    project = project_factor_approx

    @property
    def mean_field(self) -> MeanField:
        return MeanField.prod(
            {v: 1. for v in self.all_variables},
            *self._factor_mean_field.values())

    @property
    def variable_factor_message(self
    ) -> Dict[Variable, Dict[Factor, AbstractMessage]]:
        variable_factor_message = {
            v: {} for v in self.all_variables}
        for factor, meanfield in self.factor_mean_field.items():
            for v, message in meanfield.items():
                variable_factor_message[v][factor] = message

        return variable_factor_message

    @property
    def variable_messages(self) -> Dict[Variable, List[AbstractMessage]]:
        variable_messages = {
            v: [] for v in self.all_variables}
        for meanfield in self.factor_mean_field.values():
            for v, message in meanfield.items():
                variable_messages[v].append(message)
        
        return variable_messages

    @property
    def variable_evidence(self) -> Dict[Variable, np.ndarray]:
        return {
            v: AbstractMessage.log_normalisation(*ms)
            for v, ms in self.variable_messages.items()}

    @property 
    def factor_evidence(self) -> Dict[Factor, np.ndarray]:
        return {
            factor: meanfield.log_norm 
            for factor, meanfield in self.factor_mean_field.items()}
    
    @property
    def log_evidence(self):
        """
        Calculates evidence for the EP approximation

        Evidence for a variable, x_i,

        Zᵢ = ∫ ∏ₐ m_{a → i} (xᵢ) dxᵢ

        Evidence for a factor, f_a,

                ∫ ∏_{j ∈ a} m_{a → i} (xᵢ) fₐ(xₐ) dxₐ
        Zₐ = -----------------------------------------
                             ∏_{j ∈ a} Zⱼ

        Evidence for model

        Z = ∏ᵢ Zᵢ ∏ₐ Zₐ
        """
        variable_evidence = {
            v: np.sum(Zi) for v, Zi in self.variable_evidence.items()}
        factor_evidence = sum(
            np.sum(meanfield.log_norm)
             - sum(variable_evidence[v] for v in factor.all_variables)
            for factor, meanfield in self.factor_mean_field.items()
        )
        return factor_evidence + sum(variable_evidence.values())

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}({self.factor_graph}, {self.factor_mean_field})"

    def __call__(self, **kwargs: np.ndarray) -> np.ndarray:
        return self.mean_field(**kwargs)

    @property
    def is_valid(self) -> bool:
        return all(
            mean_field.is_valid for mean_field in self.factor_mean_field.values())


class AbstractFactorOptimiser(ABC):
    @abstractmethod
    def optimise(
            self, 
            factor: Factor, 
            model_approx: EPMeanField, 
            status: Optional[Status] = None
    ) -> Tuple[EPMeanField, Status]:
        pass

EPCallBack = Callable[[Factor, EPMeanField, Status], bool]

class EPOptimiser:
    """
    """
    def __init__(
            self, 
            factor_graph: FactorGraph,
            default_optimiser: AbstractFactorOptimiser,
            factor_optimisers: Dict[Factor, AbstractFactorOptimiser] = {},
            callback: Optional[EPCallBack] = None):

        self.factor_graph = factor_graph
        self.factors = \
            self.factor_graph.factors if factor_order is None else factor_order
        self.factor_optimisers = {
            factor_optimisers.get(factor, default_optimiser)
            for factor in self.factors}
        self.callback = callback

    def model_step(
            self, model_approx:EPMeanField, status=Optional[Status]
            ) -> EPMeanField:
        new_approx = model_approx
        for factor, optimiser in self.factor_optimisers.items():
            new_approx, status = optimiser.optimise(
                factor, new_approx, status=status)
            if self.callback is not None:
                stop = self.callback(factor, new_approx, status)
                if stop:
                    break

        return new_approx, status

        