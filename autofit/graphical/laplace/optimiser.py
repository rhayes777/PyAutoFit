from typing import Optional, Dict, Tuple, Any

from autofit.graphical.expectation_propagation import (
    EPMeanField,
    AbstractFactorOptimiser,
    # MeanField,
)
from autofit.graphical.mean_field import MeanField
from autofit.graphical.factor_graphs import AbstractNode
from autofit.graphical.laplace import newton


def make_posdef_hessian(mean_field, variables):
    return MeanField.precision(mean_field, variables)


class LaplaceOptimiser:
    def __init__(
        self,
        make_hessian=make_posdef_hessian,
        search_direction=newton.newton_direction,
        calc_line_search=newton.line_search,
        quasi_newton_update=newton.bfgs_update,
        stop_conditions=newton.stop_conditions,
        make_det_hessian=None,
        max_iter=100,
        hessian_kws: Optional[Dict[str, Any]] = None,
        det_hessian_kws: Optional[Dict[str, Any]] = None,
        search_direction_kws: Optional[Dict[str, Any]] = None,
        line_search_kws: Optional[Dict[str, Any]] = None,
        quasi_newton_kws: Optional[Dict[str, Any]] = None,
        stop_kws: Optional[Dict[str, Any]] = None,
    ):
        self.make_hessian = make_hessian
        self.make_det_hessian = make_det_hessian or make_hessian
        self.search_direction = search_direction
        self.calc_line_search = calc_line_search
        self.quasi_newton_update = quasi_newton_update
        self.stop_conditions = stop_conditions

        self.max_iter = max_iter

        self.hessian_kws = hessian_kws or {}
        self.det_hessian_kws = det_hessian_kws or hessian_kws or {}
        self.search_direction_kws = search_direction_kws or {}
        self.line_search_kws = line_search_kws or {}
        self.quasi_newton_kws = quasi_newton_kws or {}
        self.stop_kws = stop_kws or {}

    @property
    def default_kws(self):
        return dict(
            max_iter=self.max_iter,
            search_direction=self.search_direction,
            calc_line_search=self.calc_line_search,
            quasi_newton_update=self.quasi_newton_update,
            stop_conditions=self.stop_conditions,
            search_direction_kws=self.search_direction_kws,
            line_search_kws=self.line_search_kws,
            quasi_newton_kws=self.quasi_newton_kws,
            stop_kws=self.stop_kws,
        )

    def prepare_state(self, factor_approx, mean_field=None) -> newton.OptimisationState:
        mean_field = mean_field or factor_approx.model_dist

        free_variables = factor_approx.free_variables
        det_variables = factor_approx.deterministic_variables

        hessian = self.make_hessian(mean_field, free_variables, **self.hessian_kws)
        if det_variables:
            det_hessian = self.make_hessian(mean_field, det_variables)
        else:
            det_hessian = None

        state = newton.OptimisationState(
            factor_approx,
            factor_approx.func_jacobian,
            MeanField.mean.fget(mean_field),
            hessian,
            det_hessian,
        )

        return state

    def optimise_state(
        self, state, old_state=None, **kwargs
    ) -> Tuple[bool, newton.OptimisationState, str]:
        kws = {**self.default_kws, **kwargs}
        return newton.optimise_quasi_newton(state, old_state, **kws)

    def optimise_approx(
        self, factor_approx, mean_field=None, **kwargs
    ) -> Tuple[bool, newton.OptimisationState, str]:
        state = self.prepare_state(factor_approx, mean_field)

        success, next_state, message = self.optimise_state(state, **kwargs)
        return success, next_state, state, message

    def optimise(self, factor_approx, mean_field=None, **kwargs) -> "MeanField":
        # TODO implement...
        pass


# class LaplaceOptimiser(AbstractFactorOptimiser):
#     def
