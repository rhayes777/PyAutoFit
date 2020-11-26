from collections import ChainMap
from itertools import chain
from functools import reduce
from typing import (
    Dict, Tuple, Optional, NamedTuple, Iterator, List, Union
)
from functools import partial

import numpy as np

from autofit.graphical.factor_graphs import \
    Factor, AbstractNode, FactorGraph, FactorValue, JacobianValue
from autofit.graphical.messages import \
    AbstractMessage, FixedMessage, map_dists
from autofit.graphical.utils import \
    prod, add_arrays, OptResult, Status, aggregate, diag, Axis
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
        q_cavity = factor_approx.cavity_dist.get(v)
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
    """For a factor with multiple variables, this class represents the 
    the mean field approximation to that factor, 

    f(x₁, x₂, x₃) = q(x₁, x₂, x₃) = q₁(x₁) q₂(x₂) q₃(x₃)

    Internally these variables approximations are stored in a 
    dictionary with the variables as keys and the message or 
    variable distribution as values


    Methods
    -------
    keys()
        returns the variables of the mean_field

    logpdf({x₁: x1, x₂: x2, x₃: x3})
        returns the q(x₁, x₂, x₃), axis defines the axes over which
        to reduce the return, if the meanfield is duplicated over multiple
        plates. 

    logpdf_gradient({x₁: x1, x₂: x2, x₃: x3})
        returns the q(x₁, x₂, x₃) and the gradients for each input. 
        to save memory the gradients are always the shape of the input
        values (i.e. this does not calculate the Jacobian)
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
        var_names = self.name_variable_dict
        return self.logpdf(
            {var_names[k]: value for k, value in kwargs.items()})

    @property
    def mean(self):
        return {v: dist.mean for v, dist in self.items()}

    @property 
    def variance(self):
        return {v: dist.variance for v, dist in self.items()}

    @property 
    def scale(self):
        return {v: dist.scale for v, dist in self.items()}
    
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
        reprdict = "{\n" + "\n".join(
            "  {}: {}".format(k, v) for k, v in self.items()) + "\n  }"
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

    def log_normalisation(self, other: 'MeanField') -> float:
        return sum(
            np.sum(dist.log_normalisation(other[v]))
            for v, dist in self.items()
        )

    def update_invalid(self, other: "MeanField") -> "MeanField":
        mean_field = {}
        for k, m in self.items():
            m2 = other.get(k)
            mean_field[k] = m.update_invalid(m2) if m2 else m

        return type(self)(mean_field, self.log_norm)

    def project_mode(self, res: OptResult):
        projection = type(self)({
            v: dist.from_mode(res.mode[v], res.hess_inv.get(v))
            for v, dist in self.items()})
        
        projection.log_norm = (
            res.log_norm - projection(res.mode, axis=None).log_value)
        return projection

    def _project_mode(
            self, 
            mode: Dict[Variable, np.ndarray],
            covar: Dict[Variable, np.ndarray], 
            fun: Optional[float] = None):
        """
        Projects the mode and covariance 
        """
        projection = MeanField({
            v: dist.from_mode(mode[v], covar.get(v))
            for v, dist in self.items()})
        if fun is not None:
            projection.log_norm = fun - projection(mode).log_value
            
        return projection

    def sample(self, n_samples=None):
        return {v: dist.sample(n_samples) for v, dist in self.items()}

    def kl(self, mean_field: 'MeanField') -> np.ndarray:
        return sum(
            np.sum(dist.kl(mean_field[k]))
            for k, dist in self.items()
        ) 

    __hash__ = Factor.__hash__ 
    
    @classmethod
    def from_dist(
        cls, 
        dist: Union[Dict[Variable, AbstractMessage], "MeanField"]
    ) -> "MeanField":
        return dist if isinstance(dist, cls) else MeanField(dist)


class FactorApproximation(AbstractNode):
    """
    This class represents the 'tilted distribution' in EP,

    When approximating a model distribution of the form,

    m(x) = ∏ₐ fₐ(xₐ)

    we can define an approximating distribution as the product of 
    factor distributions,

    q(x) = ∏ₐ qₐ(xₐ)

    the 'cavity distribution' q⁻ᵃ for a factor can be viewed as a 
    prior distribution for the factor,

    q⁻ᵃ(xₐ) = ∏_{ᵦ ≠ a} qᵦ(xᵦ)

    so the model can be approximated by the 'tilted distribution'

    q⁺ᵃ(xₐ) = fₐ(xₐ) q⁻ᵃ(xₐ)

    Parameters
    ----------
    is_valid
    factor: Factor
        fₐ(xₐ)
    cavity_dist: MeanField
        q⁻ᵃ(xₐ)
    factor_dist: MeanField
        qₐ(xₐ)
    model_dist: MeanField
        q(xₐ)

    Methods
    -------
    __call__(values={xₐ: x₀}, axis=axis)
        returns q⁺ᵃ(x₀)

    func_jacobian(values={xₐ: x₀}, variables=(xₐ,), axis=axis)
        returns q⁺ᵃ(x₀), {xₐ: dq⁺ᵃ(x₀)/dxₐ}

    project_mean_field(mean_field, delta=1., status=Status())
        for qᶠ = mean_field, finds qₐ such that qᶠₐ * q⁻ᵃ = qᶠ
        delta controls how much to change from the original factor qₐ
        so qʳₐ = (qᶠₐ)ᵟ * (qᶠₐ)¹⁻ᵟ

        returns qʳₐ, status
    """
    def __init__(
        self, 
        factor: Factor, 
        cavity_dist: MeanField, 
        factor_dist: MeanField, 
        model_dist: MeanField
    ):
        # Have to seperate FactorApproximation into two classes
        # in order to be able to redefine __new__
        self.factor = factor
        self.cavity_dist = MeanField.from_dist(cavity_dist)
        self.factor_dist = MeanField.from_dist(factor_dist)
        self.model_dist = MeanField.from_dist(model_dist)

        super().__init__(**factor._kwargs)

    @property
    def variables(self):
        return self.factor.variables

    @property
    def deterministic_variables(self):
        return self.factor.deterministic_variables

    @property
    def all_variables(self):
        return self.factor.all_variables

    @property
    def name(self):
        return f"FactorApproximation({self.factor.name})"

    @property
    def deterministic_dist(self):
        """
        the `MeanField` approximation of the deterministic variables
        """
        return MeanField({
            v: self.cavity_dist[v] for v in self.deterministic_variables})

    @property
    def is_valid(self) -> bool:
        """
        returns whether all the distributions in the factor approximation
        are valid
        """
        dists = chain(
            self.cavity_dist.values(),
            self.factor_dist.values(),
            self.model_dist.values())
        return all(d.is_valid for d in dists if isinstance(d, AbstractMessage))

    def __call__(
            self, 
            values: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        fval = self.factor(values, axis=axis)
        log_meanfield = self.cavity_dist(
            {**values, **fval.deterministic_values}, axis=axis)
        return add_arrays(fval, log_meanfield)

    def func_jacobian(
            self, 
            variable_dict: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
            **kwargs, 
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

    def project_mean_field(
            self, 
            model_dist: MeanField, 
            delta: float = 1.,
            status: Optional[Status] = None,
    ) -> "FactorApprox":
        success, messages = Status() if status is None else status

        factor_dist = (model_dist / self.cavity_dist)
        if delta < 1:
            log_norm = factor_dist.log_norm
            factor_dist = (
                factor_dist**delta * self.factor_dist**(1-delta))
            factor_dist.log_norm = (
                delta * log_norm + (1 - delta) *  self.factor_dist.log_norm)

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
