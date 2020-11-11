from collections import ChainMap
from itertools import chain
from functools import reduce
from typing import (
    Dict, Tuple, Optional, NamedTuple, Iterator, List, Union
)
from functools import partial

import numpy as np

from autofit.graphical.factor_graphs import (
    Factor, AbstractNode, FactorGraph)
from autofit.graphical.messages import FixedMessage, map_dists
from autofit.graphical.messages.abstract import AbstractMessage
from autofit.graphical.utils import (
    prod, add_arrays, OptResult, Status, FactorValue, JacobianValue, 
    aggregate, diag, Axis)
from autofit.mapper.variable import Variable

VariableFactorDist = Dict[str, Dict[Factor, AbstractMessage]]
Projection = Dict[str, AbstractMessage]


def project_on_to_factor_approx(
        factor_approx: "FactorApproximation",
        model_dist: Dict[str, AbstractMessage],
        delta: float = 1.,
        status: Optional[Status] = None
) -> Tuple["FactorApproximation", Status]:
    """
    For a passed FactorApproximation this calculates the 
    factor messages such that 
    
    model_dist = factor_dist * cavity_dist
    """
    success, messages = Status() if status is None else status
    assert 0 < delta <= 1

    factor_projection = {}
#     log_norm = 0.
    for v, q_fit in model_dist.items():
        q_cavity = factor_approx.all_cavity_dist.get(v)
        if isinstance(q_fit, FixedMessage):
            factor_projection[v] = q_fit
        elif q_fit.is_valid:
            if q_cavity:
                q_f0 = factor_approx.factor_dist[v]
                q_f1 = (q_fit / q_cavity)
            else:
                # In the case that q_cavity does not exist the model fit
                # equals the factor approximation
                q_f1 = q_fit

            # weighted update
            if delta != 1:
                q_f1 = (q_f1 ** delta).sum_natural_parameters(q_f0 ** (1 - delta))

            if not q_f1.is_valid:
                # partial updating of values
                q_f1 = q_f1.update_invalid(q_f0)
                messages += (
                    f"factor projection for {v} with {factor_approx.factor} contained "
                    "invalid values",)

            if not q_f1.is_valid:
                success = False
                messages += (
                    f"factor projection for {v} with {factor_approx.factor} is invalid",)

            factor_projection[v] = q_f1

            #             if isinstance(q_cavity, AbstractMessage):
#             log_norm += (np.sum(q_fit.log_norm)
#                          - np.sum(q_f1.log_normalisation(q_cavity)))
        else:
            success = False
            messages += (
                f"model projection for {v} with {factor_approx.factor} is invalid",)

            factor_projection[v] = factor_approx.factor_dist[v]
            q_model = (q_fit ** delta).sum_natural_parameters(
                factor_approx.model_dist[v] ** (1 - delta))
            if q_model.is_valid:
                model_dist[v] = q_model

    projection = FactorApproximation(
        factor_approx.factor,
        factor_approx.cavity_dist, 
        factor_dist=MeanField(factor_projection),
        model_dist=MeanField(model_dist),
#         log_norm=log_norm
    )
    status = Status(success, messages)

    return projection, status

    
class MeanField(Dict[Variable, AbstractMessage], Factor):
    """
    """
    def __init__(
            self, 
            dists: Dict[Variable, AbstractMessage], 
            log_norm: np.ndarray = 0.):
        dict.__init__(self, dists)
        Factor.__init__(
            self, self._logpdf, **{v.name: v for v in dists})

        if isinstance(dists, MeanField):
            self.log_norm = dists.log_norm
        else:
            self.log_norm = log_norm

    pop = dict.pop 
    values = dict.values 
    items = dict.items
    __getitem__ = dict.__getitem__
        
    def _logpdf(self, **kwargs: np.ndarray) -> np.ndarray:
        var_names = self.variable_names
        return self.logpdf(
            {var_names[k]: value for k, value in kwargs.items()})
    
    def logpdf(
            self, 
            values: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> np.ndarray:
        """Calculates the logpdf of the passed values for messages 

        the result is broadcast to the appropriate shape given the variable
        plates
        """
        return reduce(
            add_arrays, 
            (aggregate(
                self._broadcast(
                    self._variable_plates[v], m.logpdf(values[v])),
                axis = axis)
            for v, m in self.items())
        )

    def __call__(
            self, 
            values: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        return FactorValue(self.logpdf(values, axis=axis), {})

    def logpdf_gradient(
            self, 
            values: Dict[Variable, np.ndarray], 
            axis: Axis = False, 
            **kwargs):
        logl = 0
        gradl = {}
        for v, m in self.items():
            lv, gradl[v] = m.logpdf_gradient(values[v])
            lv = aggregate(
                self._broadcast(self._variable_plates[v], lv),
                axis = axis)
            logl = add_arrays(logl, lv)

        return logl, gradl

    def logpdf_gradient_hessian(
            self, 
            values: Dict[Variable, np.ndarray], 
            axis: Optional[Union[bool, int, Tuple[int, ...]]] = False, 
            **kwargs):
        logl = 0.
        gradl = {}
        hessl = {}
        for v, m in self.items():
            lv, gradl[v], hessl[v] = m.logpdf_gradient_hessian(values[v])
            lv = aggregate(
                self._broadcast(self._variable_plates[v], lv),
                axis = axis)
            logl = add_arrays(logl, lv)

        return logl, gradl, hessl

    def __repr__(self):
        reprdict = dict.__repr__(self)
        classname = (type(self).__name__)
        return f"{classname}({reprdict}, log_norm={self.log_norm})"
    
    @property
    def is_valid(self):
        return all(d.is_valid for d in self.values())
    
    def prod(self, *approxs: 'MeanField') -> 'MeanField':
        dists = (
            (k, prod((m.get(k, 1.) for m in approxs), m))
            for k, m in self.items())
        return MeanField({
            k: m for k, m in dists if isinstance(m, AbstractMessage)})

    __mul__ = prod
    
    def __truediv__(self, other: 'MeanField') -> 'MeanField':
        return type(self)({
            k: m / other.get(k, 1.) for k, m in self.items()},
            self.log_norm - other.log_norm)

    def __pow__(self, other: float) -> 'MeanField':
        return type(self)({
            k: m**other for k, m in self.items()},
            self.log_norm * other)

    def update_invalid(self, other: "MeanField") -> "MeanField":
        mean_field = {}
        for k, m in self.items():
            m2 = other.get(k)
            mean_field[k] = m.update_invalid(m2) if m2 else m

        return type(self)(mean_field, self.log_norm)

    def project_mode(self, res: OptResult):
        projection = type(self)({
            v: dist.from_mode(res.mode[v], res.inv_hessian.get(v))
            for v, dist in self.items()})
        
        projection.log_norm = res.log_norm - projection(res.mode).log_value
        return projection

    def _project_mode(
            self, 
            mode: Dict[Variable, np.ndarray],
            covar: Dict[Variable, np.ndarray], 
            fun: Optional[float] = None):
        """
        Projects mode and covar
        """
        projection = MeanField({
            v: dist.from_mode(mode[v], covar.get(v))
            for v, dist in self.items()})
        if fun is not None:
            projection.log_norm = fun - projection(mode).log_value
            
        return projection

    def sample(self, n_samples=None):
        return {v: dist.sample(n_samples) for v, dist in self.items()}

    __hash__ = Factor.__hash__ 
    
    @classmethod
    def as_meanfield(
        cls, 
        dist: Union[Dict[Variable, AbstractMessage], "MeanField"]
    ) -> "MeanField":
        return dist if isinstance(dist, cls) else MeanField(dist)


class _FactorApproximation(NamedTuple):
    factor: Factor
    cavity_dist: MeanField
    factor_dist: MeanField
    model_dist: MeanField

class FactorApproximation(_FactorApproximation):
    def __new__(
        cls, 
        factor, 
        cavity_dist, 
        factor_dist, 
        model_dist):
        # Have to seperate FactorApproximation into two classes
        # in order to be able to redefine __new__
        return super().__new__(
            cls, factor, 
            MeanField.as_meanfield(cavity_dist),
            MeanField.as_meanfield(factor_dist),
            MeanField.as_meanfield(model_dist))

    @property
    def deterministic_variables(self):
        return self.factor.deterministic_variables

    @property
    def deterministic_dist(self):
        return MeanField({
            v: self.cavity_dist[v] 
            for v in self.deterministic_variables})

    @property
    def all_cavity_dist(self):
        return ChainMap(
            self.cavity_dist,
            self.deterministic_dist
        )

    @property
    def is_valid(self) -> bool:
        dists = chain(
            self.cavity_dist.values(),
            self.factor_dist.values(),
            self.model_dist.values())
        return all(d.is_valid for d in dists if isinstance(d, AbstractMessage))

    def __call__(
            self, 
            values: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> np.ndarray:
        fval = self.factor(values, axis=axis)
        log_meanfield = self.cavity_dist(
            {**values, **fval.deterministic_values}, axis=axis)
        return add_arrays(fval, log_meanfield)
        # refactor as a mapreduce?
        # for res in chain(map_dists(self.cavity_dist, kwargs),
        #                  map_dists(self.deterministic_dist, det_vars)):
        # for res in map_dists(self.cavity_dist, {**det_vars, **kwargs}):
        #     # need to add the arrays whilst preserving the sum
        #     log_result = add_arrays(
        #         log_result, self.factor.broadcast_variable(*res))

        # return log_result

    def func_jacobian(
            self, 
            variable_dict: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
    ) -> Tuple[FactorValue, JacobianValue]:

        if axis is not None:
            raise NotImplementedError(
                "FactorApproximation.func_jacobian has not implemeted "
                f"axis={axis}, try axis=None")

        if variables is None:
            fixed_variables = set(
                v for v, m in self.model_dist.items() 
                if isinstance(m, FixedMessage))
            variables = self.factor.variables - fixed_variables

        fval, fjac = self.factor.func_jacobian(
            variable_dict, variables, axis=axis, 
            _calc_deterministic=_calc_deterministic)

        values = {**variable_dict, **fval.deterministic_values}
        var_sizes = {v: np.size(x) for v, x in values.items()}
        var_shapes = {v: np.shape(x) for v, x in values.items()}
        log_cavity, grad_cavity = self.cavity_dist.logpdf_gradient(
            values, axis=axis)

        logl = fval + log_cavity

        for v in fjac:
            fjac[v] += grad_cavity[v]

        # Update gradients using jacobians of deterministic variables
        # TODO: Should add logic to account for pullbacks for 
        #       AD frameworks e.g. Zygote.jl
        for var, grad in fjac.items():
            for det, jac in grad.deterministic_values.items():
                det_grad = grad_cavity[det].ravel()
                g = jac.reshape(var_sizes[det], var_sizes[var])
                fjac[var] += det_grad.dot(g).reshape(var_shapes[var])
        
        return logl, fjac

    project_on_to_factor_approx = project_on_to_factor_approx

    def project_mean_field(
            self, 
            model_dist: MeanField, 
            delta: float = 1.,
            status: Optional[Status] = None,
    ) -> "FactorApprox":
        success, messages = Status() if status is None else status

        factor_dist = (model_dist / self.cavity_dist)
        if delta < 1:
            factor_dist = (
                factor_dist**delta * self.factor_dist**(1-delta))
        
        if not factor_dist.is_valid:
            success = False
            messages += (
                f"model projection for {self} is invalid",)
            factor_dist = factor_dist.update_invalid(self.factor_dist)

        new_approx = FactorApproximation(
            self.factor,
            self.cavity_dist, 
            factor_dist=factor_dist,
            model_dist=model_dist,
        )
        return new_approx, Status(success, messages)

    project = project_mean_field

    def __repr__(self):
        # TODO make this nicer
        return f"{type(self).__name__}({self.factor}, ...)"

    
class MeanFieldApproximation:
    '''
    TODO: rename this EP approximation
    '''
    def __init__(
            self,
            factor_graph: FactorGraph,
            variable_factor_dist: VariableFactorDist,
            factor_evidence: Optional[Dict[Factor, float]] = None
    ):
        self._factor_graph = factor_graph
        self._variable_factor_dist = variable_factor_dist
        if factor_evidence is None:
            factor_evidence = {f: 0. for f in self.factor_graph.factors}

        self._factor_evidence = factor_evidence

    def __getitem__(self, item):
        if isinstance(item, Variable):
            return self.approx[item]
        elif isinstance(item, Factor):
            return self.factor_approximation(item)
        else:
            raise TypeError(
                f"type passed {(type(item))} is not `Variable` or `Factor`")

    @classmethod
    def from_approx_dists(
            cls,
            factor_graph: FactorGraph,
            approx_dists: Dict[Variable, AbstractMessage],
            factor_evidence: Optional[Dict[Factor, float]] = None,
    ) -> "MeanFieldApproximation":
        variable_factor_dist = {}
        for factor, variables in factor_graph.factor_all_variables.items():
            for variable in variables:
                variable_factor_dist.setdefault(
                    variable, {}
                ).setdefault(
                    factor, approx_dists[variable]
                )

        return cls(
            factor_graph,
            variable_factor_dist,
            factor_evidence=factor_evidence
        )

    @classmethod
    def from_kws(
            cls,
            factor_graph: FactorGraph,
            approx_dists: Dict[Variable, AbstractMessage],
            factor_evidence: Optional[Dict[Factor, float]] = None,
    ) -> "MeanFieldApproximation":
        return cls.from_approx_dists(
            factor_graph=factor_graph,
            approx_dists=approx_dists,
            factor_evidence=factor_evidence
        )

    def project(self, projection: FactorApproximation,
                status: Optional[Status] = None
                ) -> Tuple["MeanFieldApproximation", Status]:
        """
        """
        success, messages = Status() if status is None else status

        factor = projection.factor
        factor_projection = projection.factor_dist
        factor_evidence = self._factor_evidence.copy()
        variable_factor_dist = {
            variable: factor_dist.copy()
            for variable, factor_dist in self._variable_factor_dist.items()}

        if success and projection.is_valid:
            for v, dist in factor_projection.items():
                variable_factor_dist[v][factor] = dist

            # factor_evidence[factor] = projection.log_norm
        else:
            messages += f"projection for {factor} is invalid",
            success = False

        status = Status(success, messages)
        approx = type(self)(factor_graph=self._factor_graph,
                            variable_factor_dist=variable_factor_dist,
                            factor_evidence=factor_evidence)
        return approx, status

    @property
    def factor_graph(self) -> FactorGraph:
        return self._factor_graph

    def _variable_cavity_dist(self, variable: str,
                              cavity_factor: Factor
                              ) -> Optional[AbstractMessage]:
        dists = [dist for factor, dist in
                 self._variable_factor_dist[variable].items()
                 if factor != cavity_factor]
        if dists:
            return prod(dists)
        return None

    def factor_approximation(self, factor: Factor) -> FactorApproximation:
        var_cavity = ((v, self._variable_cavity_dist(v, factor))
                      for v in factor.all_variables)
        # Some variables may only appear once in the factor graph
        # in this case they might not have a cavity distribution
        var_cavity = MeanField({
            v: dist for v, dist in var_cavity if dist}, 
            log_norm=0.)
        # det_cavity = {
        #     v: self._variable_cavity_dist(v, factor)
        #     for v in factor.deterministic_variables}
        factor_dist = MeanField({
            v: self._variable_factor_dist[v][factor]
            for v in factor.all_variables})
        model_dist = MeanField({
            v: self[v] for v in factor.all_variables})

        return FactorApproximation(
            factor, var_cavity, factor_dist, model_dist)

    def __repr__(self) -> str:
        name = type(self).__name__
        fac, varfacdist = self._factor_graph, self._variable_factor_dist
        return f"{name}({fac}, {varfacdist})"

    @property
    def approx(self) -> Dict[Variable, AbstractMessage]:
        return {
            v: prod(factors.values())
            for v, factors
            in self._variable_factor_dist.items()
        }

    def __call__(self, **kwargs: Dict[str, np.ndarray]) -> np.ndarray:
        return sum(
            prod(factors.values()).logpdf(kwargs[v])
            for v, factors in self._variable_factor_dist.items())

    @property
    def is_valid(self) -> bool:
        return (
                all(dist.is_valid for factors in
                    self._variable_factor_dist.values()
                    for dist in factors.values())
                and all(dist.is_valid for dist in self.approx.values()))

