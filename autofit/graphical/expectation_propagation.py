
from abc import ABC, abstractmethod
from collections import defaultdict 
from itertools import count
from typing import (
    Dict, Tuple, Optional, NamedTuple, Iterator, List,
    Callable
)

import numpy as np

from autofit.graphical.messages.abstract import AbstractMessage

from autofit.graphical.factor_graphs import (
    Factor, AbstractNode, FactorGraph)
from autofit.mapper.variable import Variable
from autofit.graphical.utils import Status, cached_property
from autofit.graphical.mean_field import MeanField, FactorApproximation 


class EPMeanField(AbstractNode):
    '''
    this class encode the EP mean-field approximation to a factor graph

    
    Attributes
    ----------
    factor_graph: FactorGraph
        the base factor graph being approximated

    factor_mean_field: Dict[Factor, MeanField]
        the mean-field approximation for each factor in the factor graph

    mean_field: MeanField
        the mean-field approximation of the full factor graph
        i.e. the product of the factor mean-field approximations

    variables: Set[Variable]
        the variables of the approximation

    deterministic_variables: Set[Variable]
        the deterministic variables

    log_evidence: float
        the approximate log evidence of the approximation

    is_valid: bool
        returns whether the factor mean-field approximations are all valid

    Methods
    -------
    from_approx_dists(factor_graph, approx_dists)
        create a EPMeanField object from the passed factor_graph
        using approx_dists to initialise the factor mean-field approximations

    factor_approximation(factor)
        create the FactorApproximation for the factor

    project_factor_approx(factor_approximation)
        given the passed FactorApproximation, return a new `EPMeanField`
        object encoding the updated mean-field approximation
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

    # @property
    # def variable_names(self) -> Dict[str, Variable]: 
    #     return self.factor_graph.variable_names
 
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
                v: approx_dists[v] for v in factor.all_variables})
            for factor in factor_graph.factors}

        return cls(
            factor_graph,
            factor_mean_field)

    from_kws = from_approx_dists
    
    def factor_approximation(self, factor: Factor) -> FactorApproximation:
        factor_mean_field = self._factor_mean_field.copy()
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
            for v, ms in self.variable_messages.items()
        }

    @property 
    def factor_evidence(self) -> Dict[Factor, np.ndarray]:
        return {
            factor: meanfield.log_norm 
            for factor, meanfield in self.factor_mean_field.items()
        }
    
    @property
    def log_evidence(self):
        """
        Calculates evidence for the EP approximation

        Evidence for a variable, xᵢ,

        Zᵢ = ∫ ∏ₐ m_{a → i} (xᵢ) dxᵢ

        Evidence for a factor, f_a,

                ∫ ∏_{j ∈ a} m_{i → a} (xᵢ) fₐ(xₐ) dxₐ
        Zₐ = -----------------------------------------
                             ∏_{j ∈ a} Zⱼ

        Evidence for model

        Z = ∏ᵢ Zᵢ ∏ₐ Zₐ
        """
        variable_evidence = {
            v: np.sum(logz) for v, logz in self.variable_evidence.items()}
        factor_evidence = sum(
            np.sum(meanfield.log_norm)
            - sum(variable_evidence[v] for v in factor.all_variables)
            for factor, meanfield in self.factor_mean_field.items()
        )
        return factor_evidence + sum(variable_evidence.values())

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            f"{clsname}({self.factor_graph}, "
            f"log_evidence={self.log_evidence})")

    def __call__(self, **kwargs: np.ndarray) -> np.ndarray:
        return self.mean_field(**kwargs)

    @property
    def is_valid(self) -> bool:
        return all(mean_field.is_valid 
                   for mean_field in self.factor_mean_field.values())


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

class EPHistory:
    def __init__(
            self, 
            callbacks: Tuple[EPCallBack, ...] = (),
            kl_tol=1e-6,
            evidence_tol=None):
        self._callbacks = callbacks
        self.history = {}
        self.statuses = {}
        self.factor_count = defaultdict(count)

        self.kl_tol = kl_tol 
        self.evidence_tol = evidence_tol

    def __call__(
            self, 
            factor: Factor, 
            approx: EPMeanField, 
            status: Status = Status()
    ) -> bool:
        i = next(self.factor_count[factor])
        self.history[i, factor] = approx 
        self.statuses[i, factor] = status

        stop = any([
            callback(factor, approx, status) for callback in self._callbacks
        ])
        if stop:
            return True
        elif i:
            last_approx = self.history[i-1, factor]
            return self._check_convergence(approx, last_approx)
        
        return False

    def _kl_convergence(
            self, 
            approx: EPMeanField, 
            last_approx: EPMeanField, 
    ) -> bool:
        return approx.mean_field.kl(last_approx.mean_field) < self.kl_tol

    def _evidence_convergence(
            self, 
            approx: EPMeanField, 
            last_approx: EPMeanField, 
    ) -> bool:
        last_evidence = last_approx.log_evidence
        evidence = approx.log_evidence 
        if last_evidence > evidence:
            # todo print warning?
            return False
            
        return evidence - last_evidence < self.evidence_tol

    def _check_convergence(
            self, 
            approx: EPMeanField, 
            last_approx: EPMeanField, 
    ) -> bool:
        stop = False
        if self.kl_tol:
            stop = stop or self._kl_convergence(approx, last_approx)

        if self.evidence_tol:
            stop = stop or self._evidence_convergence(approx, last_approx)

        return stop



class EPOptimiser:
    """
    """
    def __init__(
            self, 
            factor_graph: FactorGraph,
            default_optimiser: AbstractFactorOptimiser = None,
            factor_optimisers: Dict[Factor, AbstractFactorOptimiser] = {},
            callback: Optional[EPCallBack] = None,
            factor_order: Optional[List[Factor]] = None):

        self.factor_graph = factor_graph
        self.factors = \
            self.factor_graph.factors if factor_order is None else factor_order
        
        if default_optimiser is None:
            self.factor_optimisers = factor_optimisers
            missing = set(self.factors) - self.factor_optimisers.keys()
            if missing:
                raise(ValueError(
                    f"missing optimisers for {missing}, "
                    "pass a default_optimiser or add missing optimsers"
                    ))
        else:
            self.factor_optimisers = {
                factor: factor_optimisers.get(factor, default_optimiser)
                for factor in self.factors}

        self.callback = callback or EPHistory()

    def run(
            self, 
            model_approx: EPMeanField, 
            max_steps=100,
    ) -> EPMeanField:
        for _ in range(max_steps):
            for factor, optimiser in self.factor_optimisers.items():
                model_approx, status = optimiser.optimise(factor, model_approx)
                if self.callback(factor, model_approx, status):
                    break # callback controls convergence
            else: # If no break do next iteration
                continue
            break  # stop iterations

        return model_approx

        