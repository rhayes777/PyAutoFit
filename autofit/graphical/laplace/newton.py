from typing import Optional, Callable, Dict, NamedTuple, Any, Generator

import numpy as np

from autofit.graphical.factor_graphs.abstract import (
    FactorValue,
    FactorInterface,
    FactorJacobianInterface,
)
from autofit.mapper.variable_operator import VariableData, AbstractVariableOperator

from autofit.graphical.laplace.line_search import line_search, LineSearchFunc

Factor = Callable[[VariableData], FactorValue]


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
    zkdk = zk.dot(dk)
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
    ykTdk = -yk.dot(dk)

    Bdk = Bk.dot(dk)
    dkTBdk = VariableData.dot(Bdk, dk)

    Bknew = Bk.update(
        (yk, VariableData(yk).div(ykTdk)), (Bdk, VariableData(Bdk).div(dkTBdk))
    )
    return Bknew


def quasi_deterministic_update(
    fk: FactorValue,
    fk1: FactorValue,
    xk: VariableData,
    xk1: VariableData,
    gk: VariableData,
    gk1: VariableData,
    Bxk: AbstractVariableOperator,
    Bzk: AbstractVariableOperator,
):
    dk = VariableData.sub(xk, xk1)
    zk = VariableData.sub(fk.deterministic_values, fk1.deterministic_values)
    zkTzk2 = zk.dot(zk) ** 2
    alpha = (zk.dot(Bzk.dot(zk)) - dk.dot(Bxk.dot(dk))) / zkTzk2
    return Bzk.update((zk, alpha * zk))


## Newton step


class NextStep(NamedTuple):
    fval: FactorValue
    xval: VariableData
    gval: VariableData
    f_count: int
    g_count: int


class NextQuasiStep(NamedTuple):
    fval: FactorValue
    xval: VariableData
    gval: VariableData
    B: AbstractVariableOperator
    det_B: Optional[AbstractVariableOperator]
    f_count: int
    g_count: int


def take_step(
    factor: FactorInterface,
    factor_jacobian: FactorJacobianInterface,
    xk: VariableData,
    gk: Optional[VariableData] = None,
    bk: Optional[AbstractVariableOperator] = None,
    fk: Optional[FactorValue] = None,
    fk_m1: Optional[FactorValue] = None,
    *,
    args: tuple = (),
    search_direction=newton_direction,
    calc_line_search: LineSearchFunc = line_search,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    f_count=0,
    g_count=0,
):
    ls_kws = line_search_kws or {}
    if gk is None:
        _, gk = factor_jacobian(xk, *args)
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
        f_count=f_count,
        g_count=g_count,
        **ls_kws,
    )
    x_next = xk + VariableData.mul(pk, ls.alpha)
    return NextStep(ls.fval, x_next, ls.gval, ls.f_count, ls.g_count)


def take_quasi_newton_step(
    factor: FactorInterface,
    factor_jacobian: FactorJacobianInterface,
    xk: VariableData,
    gk: Optional[VariableData] = None,
    Bk: Optional[AbstractVariableOperator] = None,
    fk: Optional[FactorValue] = None,
    fk_m1: Optional[FactorValue] = None,
    det_Bk: Optional[AbstractVariableOperator] = None,
    *,
    args: tuple = (),
    search_direction=newton_direction,
    calc_line_search: LineSearchFunc = line_search,
    quasi_newton_update=bfgs_update,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
    f_count=0,
    g_count=0,
) -> NextQuasiStep:
    next_step = take_step(
        factor,
        factor_jacobian,
        xk,
        gk,
        Bk,
        fk,
        fk_m1,
        args=args,
        search_direction=search_direction,
        calc_line_search=calc_line_search,
        line_search_kws=line_search_kws,
        search_direction_kws=search_direction_kws,
        f_count=f_count,
        g_count=g_count,
    )
    Bk_next = quasi_newton_update(
        next_step.fval, fk, next_step.xval, xk, next_step.gval, gk, Bk
    )
    if det_Bk:
        det_Bk = quasi_deterministic_update(
            next_step.fval,
            fk,
            next_step.xval,
            xk,
            next_step.gval,
            gk,
            Bk_next,
            det_Bk,
            **(quasi_newton_kws or {}),
        )

    return NextQuasiStep(**next_step._asdict(), B=Bk_next, det_B=det_Bk)


def iter_quasi_newton(
    factor: FactorInterface,
    factor_jacobian: FactorJacobianInterface,
    xk: VariableData,
    gk: Optional[VariableData] = None,
    Bk: Optional[AbstractVariableOperator] = None,
    fk: Optional[FactorValue] = None,
    fk_m1: Optional[FactorValue] = None,
    det_Bk: Optional[AbstractVariableOperator] = None,
    *,
    args: tuple = (),
    search_direction=newton_direction,
    calc_line_search: LineSearchFunc = line_search,
    quasi_newton_update=bfgs_update,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
    f_count: int = 0,
    g_count: int = 0,
) -> Generator[NextQuasiStep, None, None]:
    while True:
        result = take_quasi_newton_step(
            factor,
            factor_jacobian,
            xk,
            gk,
            Bk,
            fk,
            fk_m1,
            det_Bk,
            args=args,
            search_direction=search_direction,
            calc_line_search=calc_line_search,
            quasi_newton_update=quasi_newton_update,
            line_search_kws=line_search_kws,
            search_direction_kws=search_direction_kws,
            quasi_newton_kws=quasi_newton_kws,
            f_count=f_count,
            g_count=g_count,
        )
        yield result
        fk_m1 = fk
        fk, xk, gk, Bk, det_Bk = result[:5]
        f_count, g_count = result.f_count, result.g_count
