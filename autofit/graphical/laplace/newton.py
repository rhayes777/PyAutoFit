from typing import Optional, Callable, Tuple, NamedTuple

import numpy as np

from autofit.graphical.factor_graphs.abstract import FactorValue
from autofit.mapper.variable_operator import VariableData, AbstractVariableOperator

import line_search

Factor = Callable[[VariableData], FactorValue]
FactorJacobian = Callable[[VariableData], Tuple[FactorValue, VariableData]]


## get ascent direction


def gradient_ascent(
    xk: VariableData,
    gk: VariableData,
    Bk: Optional[AbstractVariableOperator] = None,
) -> VariableData:
    return VariableData(gk)


def newton_direction(
    xk: VariableData, gk: VariableData, Bk: AbstractVariableOperator
) -> VariableData:
    return Bk.ldiv(gk)


## Quasi-newton approximation


def sr1_update(
    fk: FactorValue,
    fk1: FactorValue,
    xk: VariableData,
    xk1: VariableData,
    gk: VariableData,
    gk1: VariableData,
    Bk: AbstractVariableOperator,
    mintol=1e-8,
):
    dk = VariableData.sub(xk, xk1)
    yk = VariableData.sub(gk, gk1)
    zk = yk - Bk * dk
    zkdk = zk.dot(dk).sum()
    tol = mintol * (xk.norm() ** 2).sum() ** 0.5 * (zk.norm() ** 2).sum() ** 0.5
    if zkdk > tol:
        vk = zk / np.sqrt(zkdk)
        Bknew = Bk.lowrankupdate(vk)
    elif zkdk < -tol:
        vk = zk / np.sqrt(-zkdk)
        Bknew = Bk.lowrankdowndate(vk)
    else:
        Bknew = Bk

    return Bknew


def bfgs_update(
    fk: FactorValue,
    fk1: FactorValue,
    xk: VariableData,
    xk1: VariableData,
    gk: VariableData,
    gk1: VariableData,
    Bk: AbstractVariableOperator,
):
    dk = VariableData.sub(xk, xk1)
    yk = VariableData.sub(gk, gk1)
    ykTdk = yk.dot(dk).neg().sum()

    Bdk = Bk.dot(dk)
    dkTBdk = VariableData.dot(Bdk, dk).sum()

    Bknew = Bk.update(
        (yk, VariableData(yk).div(ykTdk)), (Bdk, VariableData(Bdk).div(dkTBdk))
    )
    return Bknew


## Newton step


class NextStep(NamedTuple):
    fval: FactorValue
    xval: VariableData
    gval: VariableData
    f_count: int
    g_count: int


def take_step(
    factor: Factor,
    factor_jacobian: FactorJacobian,
    xk: VariableData,
    gk: Optional[VariableData] = None,
    bk: Optional[AbstractVariableOperator] = None,
    fk: Optional[FactorValue] = None,
    fk_m1: Optional[FactorValue] = None,
    args: tuple = (),
    search_direction=gradient_ascent,
    calc_line_search=line_search.line_search,
    # quasi_newton_update=None,
    search_direction_kws=None,
    line_search_kws=None,
    # quasi_newton_kws=None,
):
    ls_kws = line_search_kws or {}
    g_count = ls_kws.pop("g_count", 0)
    if gk is None:
        fk, gk = factor_jacobian(xk, *args)
        g_count += 1

    pk = search_direction(xk, gk, bk, **(search_direction_kws or {}))
    ls = calc_line_search(
        factor,
        factor_jacobian,
        xk,
        pk,
        gk,
        fk,
        fk_m1,
        args=args,
        g_count=g_count,
        **ls_kws
    )
    x_next = xk + VariableData.mul(pk, ls.alpha)
    return NextStep(ls.fval, x_next, ls.gval, ls.f_count, ls.g_count)
