from typing import NamedTuple
import warnings

import numpy as np
from scipy.optimize import linesearch
from scipy.optimize.optimize import _LineSearchError

from autofit.graphical.factor_graphs.abstract import FactorValue
from autofit.mapper.variable_operator import VariableData, AbstractVariableOperator


class LineSearchResult(NamedTuple):
    alpha: float
    fval: FactorValue
    gval: VariableData
    old_fval: FactorValue
    f_count: int
    g_count: int


def prepare_factor(factor, factor_jacobian, xk, pk, gk=None, args=()):
    if gk is None:
        fk, gk = factor_jacobian(xk, *args)

    pk = VariableData(pk)
    fval = [None]
    gval = [gk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return -factor(xk + s * pk, *args)

    def derphi(s):
        _, gval[0] = factor_jacobian(xk + s * pk, *args)
        gc[0] += 1
        return VariableData.dot(pk, gval[0]).neg().sum()

    derphi0 = VariableData.dot(pk, gk).neg().sum()

    vals = fval, gval, gc, fc

    return phi, derphi, derphi0, vals


def line_search_wolfe1(
    factor,
    factor_jacobian,
    xk,
    pk,
    gk=None,
    old_fval=None,
    old_old_fval=None,
    args=(),
    c1=1e-4,
    c2=0.9,
    amax=50,
    amin=1e-8,
    xtol=1e-14,
    f_count=0,
    g_count=0,
):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    phi, derphi, derphi0, (_, gval, gc, fc) = prepare_factor(
        factor, factor_jacobian, xk, pk, gk=gk, args=args
    )
    gc[0] += g_count
    fc[0] += f_count
    stp, fval, old_fval = linesearch.scalar_search_wolfe1(
        phi,
        derphi,
        old_fval,
        old_old_fval,
        derphi0,
        c1=c1,
        c2=c2,
        amax=amax,
        amin=amin,
        xtol=xtol,
    )

    return LineSearchResult(stp, fval, gval[0], old_fval, fc[0], gc[0])


def line_search_wolfe2(
    factor,
    factor_jacobian,
    xk,
    pk,
    gk=None,
    old_fval=None,
    old_old_fval=None,
    args=(),
    c1=1e-4,
    c2=0.9,
    amax=None,
    extra_condition=None,
    maxiter=10,
    f_count=0,
    g_count=0,
):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    phi, derphi, derphi0, (fval, gval, gc, fc) = prepare_factor(
        factor, factor_jacobian, xk, pk, gk=gk, args=args
    )
    gc[0] += g_count
    fc[0] += f_count

    alpha_star, phi_star, old_fval, derphi_star = linesearch.scalar_search_wolfe2(
        phi,
        derphi,
        old_fval,
        old_old_fval,
        derphi0,
        c1=c1,
        c2=c2,
        amax=amax,
        extra_condition=extra_condition,
        maxiter=maxiter,
    )

    return LineSearchResult(alpha_star, fval[0], gval[0], old_fval, fc[0], gc[0])


def line_search(
    factor, factor_jacobian, xk, pk, gk=None, old_fval=None, old_old_fval=None, **kwargs
):

    extra_condition = kwargs.pop("extra_condition", None)

    ret = line_search_wolfe1(
        factor, factor_jacobian, xk, pk, gk, old_fval, old_old_fval, **kwargs
    )

    kwargs["f_count"] = ret.f_count
    kwargs["g_count"] = ret.g_count

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", linesearch.LineSearchWarning)
            kwargs2 = {}
            for key in ("c1", "c2", "amax"):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]

            ret = line_search_wolfe2(
                factor,
                factor_jacobian,
                xk,
                pk,
                gk,
                old_fval,
                old_old_fval,
                extra_condition=extra_condition,
                **kwargs2
            )

    if ret[0] is None:
        raise _LineSearchError()

    return ret
