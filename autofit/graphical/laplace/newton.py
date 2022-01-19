from typing import Optional, Callable, Dict, Tuple, Any

import numpy as np

from autofit.graphical.factor_graphs.abstract import FactorValue
from autofit.mapper.variable_operator import VariableData

from autofit.graphical.laplace.line_search import line_search, OptimisationState


## get ascent direction


def gradient_ascent(state: OptimisationState) -> VariableData:
    return state.gradient


def newton_direction(state: OptimisationState) -> VariableData:
    return state.hessian.ldiv(state.gradient)


## Quasi-newton approximations


def sr1_update(
    state1: OptimisationState, state: OptimisationState, mintol=1e-8, **kwargs
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.variables, state.variables)
    Bk = state.hessian
    zk = yk - Bk * dk
    zkdk = zk.dot(dk)

    tol = mintol * (dk.norm() ** 2).sum() ** 0.5 * (zk.norm() ** 2).sum() ** 0.5
    if zkdk > tol:
        vk = zk / np.sqrt(zkdk)
        Bk1 = Bk.lowrankupdate(vk)
    elif zkdk < -tol:
        vk = zk / np.sqrt(-zkdk)
        Bk1 = Bk.lowrankdowndate(vk)
    else:
        Bk1 = Bk

    state1.hessian = Bk1
    return state1


def bfgs_update(
    state1: OptimisationState,
    state: OptimisationState,
    **kwargs,
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.variables, state.variables)
    Bk = state.hessian

    ykTdk = -yk.dot(dk)
    Bdk = Bk.dot(dk)
    dkTBdk = VariableData.dot(Bdk, dk)

    state1.hessian = Bk.update(
        (yk, VariableData(yk).div(ykTdk)), (Bdk, VariableData(Bdk).div(dkTBdk))
    )
    return state1


def quasi_deterministic_update(
    state1: OptimisationState,
    state: OptimisationState,
    **kwargs,
) -> OptimisationState:
    dk = VariableData.sub(state1.variables, state.variables)
    zk = VariableData.sub(
        state1.value.deterministic_values, state.value.deterministic_values
    )
    Bxk, Bzk = state.hessian, state.det_hessian
    zkTzk2 = zk.dot(zk) ** 2
    alpha = (zk.dot(Bzk.dot(zk)) - dk.dot(Bxk.dot(dk))) / zkTzk2
    state1.det_hessian = Bzk.update((zk, alpha * zk))
    return state1


## Newton step


def take_step(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    search_direction=newton_direction,
    calc_line_search=line_search,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], OptimisationState]:
    state.search_direction = search_direction(state, **(search_direction_kws or {}))
    return calc_line_search(state, old_state, **(line_search_kws or {}))


def take_quasi_newton_step(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    search_direction=newton_direction,
    calc_line_search=line_search,
    quasi_newton_update=bfgs_update,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], OptimisationState]:
    state.search_direction = search_direction(state, **(search_direction_kws or {}))
    stepsize, state1 = calc_line_search(state, old_state, **(line_search_kws or {}))
    state1 = quasi_newton_update(state1, state, **(quasi_newton_kws or {}))
    if state.det_hessian:
        state1 = quasi_deterministic_update(state1, state, **(quasi_newton_kws or {}))

    return stepsize, state1


def xtol_condition(state, old_state, xtol=1e-6, ord=None, **kwargs):
    dx = VariableData.sub(state.variables, old_state.variables).vecnorm(ord=ord)
    if dx < xtol:
        return True, f"Minimum parameter change tolerance achieved, {dx} < {xtol}"


def grad_condition(state, old_state, gtol=1e-5, ord=None, **kwargs):
    dg = VariableData.vecnorm(state.gradient, ord=ord)
    if dg < gtol:
        return True, f"Gradient tolerance achieved, {dg} < {gtol}"


def ftol_condition(state, old_state, ftol=1e-6, monotone=True, **kwargs):
    df = state.value - old_state.value
    if 0 < df < ftol:
        return True, f"Minimum function change tolerance achieved, {df} < {ftol}"
    elif monotone and df < 0:
        return False, f"factor failed to increase on next step, {df}"


def nfev_condition(state, old_state, maxfev=10000, **kwargs):
    if state.f_count > maxfev:
        return (
            True,
            f"Maximum number of function evaluations (maxfev={maxfev}) has been exceeded.",
        )


def ngev_condition(state, old_state, maxgev=10000, **kwargs):
    if state.g_count > maxgev:
        return (
            True,
            f"Maximum number of gradient evaluations (maxgev={maxgev}) has been exceeded.",
        )


def optimise_quasi_newton(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    max_iter=100,
    search_direction=newton_direction,
    calc_line_search=line_search,
    quasi_newton_update=bfgs_update,
    stop_conditions=(
        xtol_condition,
        ftol_condition,
        grad_condition,
    ),
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
    stop_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, OptimisationState, str]:

    success = False
    message = "max iterations reached"
    for i in range(max_iter):
        stepsize, state1 = take_quasi_newton_step(
            state,
            old_state,
            search_direction=search_direction,
            calc_line_search=calc_line_search,
            quasi_newton_update=quasi_newton_update,
            search_direction_kws=search_direction_kws,
            line_search_kws=line_search_kws,
            quasi_newton_kws=quasi_newton_kws,
        )
        state, old_state = state1, state

        if stepsize is None:
            message = f"abnormal termination of line search, iter={i}"
            break
        elif not np.isfinite(state.value):
            message = f"function is no longer finite, iter={i}"
            break
        else:
            for stop_condition in stop_conditions:
                stop = stop_condition(state, old_state, **(stop_kws or {}))
                if stop:
                    success, message = stop
                    break
            else:
                continue
            break

    return success, state, message
