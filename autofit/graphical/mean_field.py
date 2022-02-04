import logging
import warnings
from collections import ChainMap
from typing import Dict, Tuple, Optional, Union, Iterable

import numpy as np

from autoconf import cached_property
from autofit import exc
from autofit.graphical.factor_graphs.abstract import AbstractNode
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.jacobians import AbstractJacobian
from autofit.graphical.utils import (
    StatusFlag,
    prod,
    OptResult,
    Status,
    LogWarnings,
)
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mapper.variable import (
    Variable,
    Plate,
    VariableData,
    FactorValue,
    VariableLinearOperator,
)
from autofit.mapper.variable_operator import MatrixOperator, VariableFullOperator
from autofit.messages.abstract import AbstractMessage
from autofit.messages.fixed import FixedMessage

VariableFactorDist = Dict[str, Dict[Factor, AbstractMessage]]
Projection = Dict[str, AbstractMessage]

logger = logging.getLogger(__name__)

_log_projection_warnings = logger.debug


# Does this need to be a Factor?
class MeanField(CollectionPriorModel, Dict[Variable, AbstractMessage], Factor):
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
            plates: Optional[Tuple[Plate, ...]] = None,
            log_norm: np.ndarray = 0.0,
    ):
        dict.__init__(self, dists)
        Factor.__init__(self, self._logpdf, *dists, arg_names=[])
        CollectionPriorModel.__init__(self)

        if isinstance(dists, MeanField):
            self.log_norm = dists.log_norm
            self._plates = dists.plates
        else:
            self.log_norm = log_norm
            self._plates = self.sorted_plates if plates is None else plates

    @cached_property
    def fixed_values(self):
        return VariableData(
            {k: dist.mean for k, dist in self.items() if isinstance(dist, FixedMessage)}
        )

    @classmethod
    def from_priors(cls, priors: Iterable[Prior]) -> "MeanField":
        """
        Create a MeanField from a list of priors.

        This works because priors are a kind of variable and
        messages can be derived from priors.

        Parameters
        ----------
        priors
            A list of priors

        Returns
        -------
        A mean field
        """
        return MeanField({prior: prior for prior in priors})

    pop = dict.pop
    values = dict.values
    items = dict.items
    __getitem__ = dict.__getitem__
    __len__ = dict.__len__

    def subset(self, variables):
        cls = type(self) if isinstance(self, MeanField) else MeanField
        return cls((v, self[v]) for v in variables)

    @property
    def mean(self):
        return VariableData({v: dist.mean for v, dist in self.items()})

    @property
    def variance(self):
        return VariableData({v: dist.variance for v, dist in self.items()})

    @property
    def std(self):
        return VariableData({v: dist.std for v, dist in self.items()})

    @property
    def scale(self):
        return VariableData({v: dist.scale for v, dist in self.items()})

    def precision(self, variables=None):
        variables = variables or self.all_variables
        variances = MeanField.variance.fget(self).subset(variables)
        return VariableFullOperator.from_diagonal(variances ** -1)

    @property
    def arguments(self) -> Dict[Variable, Prior]:
        """
        Arguments that can be used to update a PriorModel
        """
        return {v: dist for v, dist in self.items()}

    def _logpdf(self, *args: np.ndarray) -> np.ndarray:
        var_names = self.name_variable_dict
        return self.logpdf(dict(zip(self.args, args)))

    def logpdf(
            self,
            values: Dict[Variable, np.ndarray],
    ) -> np.ndarray:
        """Calculates the logpdf of the passed values for messages

        the result is broadcast to the appropriate shape given the variable
        plates
        """
        return sum(np.sum(m.logpdf(values[v])) for v, m in self.items())

    def logpdf_gradient(self, values: Dict[Variable, np.ndarray], **kwargs):
        logl = 0
        gradl = {}
        for v, m in self.items():
            lv, gradl[v] = m.logpdf_gradient(values[v])
            logl += np.sum(lv)

        return logl, gradl

    def __repr__(self):
        reprdict = (
            "{\n" + "\n".join(f"  {k}: {v}" for k, v in self.items()) + "\n  }"
        )
        classname = type(self).__name__
        return f"{classname}({reprdict}, log_norm={self.log_norm})"

    @property
    def is_valid(self):
        return all(d.is_valid for d in self.values())

    def prod(self, *approxs: "MeanField") -> "MeanField":
        dists = (
            (k, prod((m.get(k, 1.0) for m in approxs), m)) for k, m in self.items()
        )
        return MeanField({k: m for k, m in dists if isinstance(m, Prior)})

    __mul__ = prod

    def __truediv__(self, other: "MeanField") -> "MeanField":
        return type(self)(
            {k: m / other.get(k, 1.0) for k, m in self.items()},
            self.log_norm - other.log_norm,
        )

    def __pow__(self, other: float) -> "MeanField":
        return type(self)(
            {k: m ** other for k, m in self.items()}, self.log_norm * other
        )

    def log_normalisation(self, other: "MeanField") -> float:
        return sum(np.sum(dist.log_normalisation(other[v])) for v, dist in self.items())

    def update_invalid(self, other: "MeanField") -> "MeanField":
        mean_field = {}
        for k, m in self.items():
            m2 = other.get(k)
            mean_field[k] = m.update_invalid(m2) if m2 else m

        return type(self)(mean_field, self.log_norm)

    def check_valid(self):
        return VariableData((v, m.check_valid()) for v, m in self.items())

    def project_mode(self, res: OptResult):
        return self.from_mode_covariance(res.mode, res.hess_inv, res.log_norm)

    def from_opt_state(self, state):
        return self.from_mode_covariance(
            state.all_parameters, state.full_hessian.inv(), state.value
        )

    def from_mode_covariance(
            self,
            mode: Dict[Variable, np.ndarray],
            covar: Dict[Variable, np.ndarray],
            fun: Optional[float] = None,
    ):
        """
        Projects the mode and covariance
        """
        mode = ChainMap(mode, self.fixed_values)
        if isinstance(covar, VariableLinearOperator):
            covar = covar.to_block(MatrixOperator).operators

        projection = MeanField(
            {
                v: self[v].from_mode(mode[v], covar.get(v), id_=self[v].id)
                for v in self.keys() & mode.keys()
            }
        )
        if fun is not None:
            projection.log_norm = fun - projection(mode).log_value

        return projection

    def sample(self, n_samples=None):
        return VariableData({v: dist.sample(n_samples) for v, dist in self.items()})

    def kl(self, mean_field: "MeanField") -> np.ndarray:
        return sum(np.sum(dist.kl(mean_field[k])) for k, dist in self.items())

    __hash__ = Factor.__hash__

    @classmethod
    def from_dist(
            cls, dist: Union[Dict[Variable, AbstractMessage], "MeanField"]
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
    __call__(values={xₐ: x₀})
        returns q⁺ᵃ(x₀)

    func_jacobian(values={xₐ: x₀}, variables=(xₐ,))
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
            model_dist: MeanField,
    ):
        # Have to seperate FactorApproximation into two classes
        # in order to be able to redefine __new__
        self.factor = factor
        self.cavity_dist = MeanField.from_dist(cavity_dist)
        self.factor_dist = MeanField.from_dist(factor_dist)
        self.model_dist = MeanField.from_dist(model_dist)

        self.fixed_values = self.factor_dist.fixed_values

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
    def mean_field(self):
        return self.model_dist

    @property
    def name(self):
        return f"FactorApproximation({self.factor.name})"

    @property
    def deterministic_dist(self):
        """
        the `MeanField` approximation of the deterministic variables
        """
        return MeanField({v: self.cavity_dist[v] for v in self.deterministic_variables})

    def __call__(
            self,
            values: Dict[Variable, np.ndarray],
    ) -> FactorValue:
        variable_dict = {**self.fixed_values, **values}
        fval = self.factor(variable_dict)
        log_meanfield = self.cavity_dist({**variable_dict, **fval.deterministic_values})
        return np.sum(fval) + np.sum(log_meanfield)

    def func_jacobian(
            self, values: Dict[Variable, np.ndarray]
    ) -> Tuple[FactorValue, AbstractJacobian]:
        raise NotImplementedError()

    def func_gradient(
            self,
            values: Dict[Variable, np.ndarray],
    ) -> Tuple[FactorValue, VariableData]:

        variable_dict = {**self.fixed_values, **values}
        fval, fjac = self.factor.func_jacobian(variable_dict)

        variable_dict.update(fval.deterministic_values)
        log_cavity, grad_cavity = self.cavity_dist.logpdf_gradient(
            {**variable_dict, **fval.deterministic_values}
        )

        logl = np.sum(fval) + np.sum(log_cavity)
        grad = fjac.grad(grad_cavity)

        return logl, grad

    def project_mean_field(
            self,
            model_dist: MeanField,
            delta: float = 1.0,
            status: Optional[Status] = None,
    ) -> Tuple["FactorApproximation", Status]:
        success, messages, _, flag = Status() if status is None else status

        updated = False
        try:
            with LogWarnings(logger=_log_projection_warnings, action='always') as caught_warnings:
                factor_dist = model_dist / self.cavity_dist
                if delta < 1:
                    log_norm = factor_dist.log_norm
                    factor_dist = factor_dist ** delta * self.factor_dist ** (1 - delta)
                    factor_dist.log_norm = (
                            delta * log_norm + (1 - delta) * self.factor_dist.log_norm
                    )

            for m in caught_warnings.messages:
                messages += (f"project_mean_field warning: {m}",)

            if not factor_dist.is_valid:
                success = False
                messages += (f"model projection for {self} is invalid",)
                factor_dist = factor_dist.update_invalid(self.factor_dist)
                # May want to check another way
                # e.g. factor_dist.check_valid().sum() / factor_dist.check_valid().size
                if factor_dist.check_valid().any():
                    updated = True

                flag = StatusFlag.BAD_PROJECTION
            else:
                updated = True

        except exc.MessageException as e:
            logger.exception(e)
            factor_dist = self.factor_dist

        new_approx = FactorApproximation(
            self.factor,
            self.cavity_dist,
            factor_dist=factor_dist,
            model_dist=model_dist,
        )
        return new_approx, Status(
            success=success, messages=messages, flag=flag, updated=updated
        )

    project = project_mean_field

    def __repr__(self):
        # TODO make this nicer
        return f"{type(self).__name__}({self.factor}, ...)"
