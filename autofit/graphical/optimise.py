from itertools import repeat
from typing import Optional, Dict, Tuple, NamedTuple, Any

import numpy as np
from scipy.optimize import minimize, OptimizeResult, least_squares

from autofit.graphical import FixedMessage
from autofit.mapper.variable import Variable
from autofit.graphical.factor_graphs import Factor
from .mean_field import FactorApproximation, MeanFieldApproximation, Status
from .utils import propagate_uncertainty, FlattenArrays


class OptResult(NamedTuple):
    mode: Dict[str, np.ndarray]
    inv_hessian: Dict[str, np.ndarray]
    result: OptimizeResult


class OptFactor:
    """
    """

    def __init__(
            self,
            factor_approx: FactorApproximation,
            kwargs: Dict[Variable, Tuple[int, ...]],
            fixed_kws: Optional[Dict[str, np.ndarray]] = None,
            bounds: Optional[Dict[str, Tuple[float, float]]] = None,
            sign: int = 1, method: str = 'L-BFGS-B',
    ):
        self.factor_approx = factor_approx
        self.param_shapes = FlattenArrays(kwargs)
        self.param_bounds = bounds
        self.free_vars = tuple(kwargs)

        self.sign = sign
        self.fixed_kws = fixed_kws
        self.method = method

        if bounds:
            # TODO check that this is correct for composite
            # distributions e.g. NormalGammaMessage
            self.bounds = [
                b for k, s in kwargs.items()
                for bound in bounds[k]
                for b in repeat(bound, np.prod(s, dtype=int))]
        else:
            self.bounds = bounds

    @classmethod
    def from_approx(
            cls,
            factor_approx: FactorApproximation,
            **kwargs
    ) -> 'OptFactor':
        fixed_kws = {}
        bounds = {}
        for v in factor_approx.factor.variables:
            dist = factor_approx.model_dist[v]
            if isinstance(dist, FixedMessage):
                fixed_kws[v] = dist.mean
            else:
                kwargs[v] = dist.shape
                bounds[v] = dist._support

        return cls(
            factor_approx,
            kwargs,
            fixed_kws=fixed_kws,
            sign=-1,
            bounds=bounds,
        )

    def __call__(self, args):
        params = self.param_shapes.unflatten(args)
        params.update(self.fixed_kws)
        return self.sign * np.sum(self.factor_approx(params))

    def _minimise(self, arrays_dict, method=None, bounds=None,
                  constraints=(), tol=None, callback=None,
                  options=None):
        x0 = self.param_shapes.flatten(arrays_dict)
        bounds = self.bounds if bounds is None else bounds
        method = self.method if method is None else method
        return minimize(
            self, x0, method=method, bounds=bounds,
            constraints=constraints, tol=tol, callback=callback,
            options=options)

    def _parse_result(self, res: OptimizeResult) -> OptResult:
        return OptResult(
            self.param_shapes.unflatten(res.x),
            self.param_shapes.unflatten(res.hess_inv.todense()),
            res)

    def minimise(self, bounds=None,
                 constraints=(), tol=None, callback=None,
                 options=None, **arrays):
        self.sign = 1
        res = self._minimise(
            bounds=bounds, constraints=constraints, tol=tol,
            callback=callback, options=options, **arrays)
        return self._parse_result(res)

    def maximise(
            self,
            arrays_dict,
            bounds=None,
            constraints=(), tol=None, callback=None,
            options=None
    ):
        self.sign = -1
        res = self._minimise(
            arrays_dict,
            bounds=bounds, constraints=constraints, tol=tol,
            callback=callback, options=options)
        self.sign = 1
        return self._parse_result(res)

    minimize = minimise
    maximize = maximise


def maximise_factor_approx(
        factor_approx: FactorApproximation, **kwargs):
    """
    """
    p0 = {
        v: kwargs.pop(v, factor_approx.model_dist[v].sample(1)[0])
        for v in factor_approx.factor.variables}
    opt = OptFactor.from_approx(factor_approx, **kwargs)
    return opt.maximise(**p0)


maximize_factor_approx = maximise_factor_approx


def find_factor_mode(
        factor_approx: FactorApproximation,
        return_cov: bool = True,
        status: Optional[Status] = None,
        min_iter: int = 2,
        **kwargs
):
    """
    """
    success, messages = Status() if status is None else status
    factor = factor_approx.factor

    opt = OptFactor.from_approx(factor_approx, **kwargs)
    free_vars = opt.free_vars
    det_vars = factor_approx.deterministic_dist
    p0 = {
        v: kwargs.pop(v, factor_approx.model_dist[v].sample(1)[0])
        for v in free_vars}

    # find the mode of the factor approximation
    res = opt.maximise(p0)
    mode = {**res.mode, **opt.fixed_kws}

    result = res.result
    success = result.success
    message = result.message.decode()
    messages += (
        "optimise.find_factor_mode: "
        f"nfev={result.nfev}, nit={result.nit}, "
        f"status={result.status}, message={message}",)
    if result.nit < min_iter:
        success = False
        # TODO implement numerical Hessian

    # find the values of the deterministic variables at the mode
    f = factor(mode)
    mode.update(f.deterministic_values)

    if return_cov:
        # Calculate covariance of deterministic variables based on
        # the free variables covariance and Jacobians
        covars = res.inv_hessian
        factor_jac = factor.jacobian(free_vars, mode)
        for det in det_vars:
            covars[det] = 0.
            for v in free_vars:
                cov = covars[v]
                jac = factor_jac.deterministic_values[det, v]
                det_cov = propagate_uncertainty(cov, jac)
                covars[det] += det_cov
    else:
        covars = {}

    return mode, covars, Status(success, messages), result


class LaplaceOptimiser:
    def __init__(
            self,
            n_iter=4,
            delta=1.
    ):
        self.history = dict()
        self.n_iter = n_iter
        self.delta = delta

    def run(self, model_approx):
        model = model_approx.factor_graph
        for i in range(self.n_iter):
            for factor in model.factors:
                # We have reduced the entire EP step into a single function
                model_approx, status = self.laplace_factor_approx(
                    model_approx,
                    factor
                )

                # save and print current approximation
                self.history[i, factor] = model_approx
        return model_approx

    def laplace_factor_approx(
            self,
            model_approx,
            factor: Factor,
            opt_kws: Optional[Dict[str, Any]] = None
    ):
        factor_approx = model_approx.factor_approximation(factor)

        opt_kws = {} if opt_kws is None else opt_kws
        mode, covars, status, _ = find_factor_mode(
            factor_approx,
            return_cov=True,
            **opt_kws
        )

        model_dist = {
            v: factor_approx.factor_dist[v].from_mode(
                mode[v],
                covars.get(v)
            )
            for v in mode
        }

        projection, status = factor_approx.project(
            model_dist,
            delta=self.delta,
            status=status
        )

        return model_approx.project(
            projection,
            status=status
        )


class LeastSquaresOpt:
    _opt_params = dict(
        jac='2-point', method='trf', ftol=1e-08,
        xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
        f_scale=1.0, diff_step=None, tr_solver=None,
        tr_options={}, jac_sparsity=None, max_nfev=None,
        verbose=0)

    def __init__(
            self,
            factor_approx: FactorApproximation,
            fixed_kws=None,
            param_bounds=None,
            opt_only=None,
            **kwargs):

        self.factor_approx = factor_approx
        self.opt_params = {**self._opt_params, **kwargs}

        param_shapes = {}
        param_bounds = {} if param_bounds is None else param_bounds
        fixed_kws = {} if fixed_kws is None else fixed_kws

        for v in factor_approx.factor.variables:
            dist = factor_approx.model_dist[v]
            if isinstance(dist, FixedMessage):
                fixed_kws[v] = dist.mean
            else:
                param_shapes[v] = dist.shape
                param_bounds[v] = dist._support

        self.fixed_kws = fixed_kws
        self.param_shapes = FlattenArrays(param_shapes)

        if opt_only is None:
            opt_only = tuple(
                v for v, d in factor_approx.all_cavity_dist.items()
                if not isinstance(d, FixedMessage)
            )

        self.opt_only = opt_only
        self.resid_means = {
            k: factor_approx.all_cavity_dist[k].mean
            for k in self.opt_only}
        self.resid_scales = {
            k: factor_approx.all_cavity_dist[k].scale
            for k in self.opt_only}
        self.resid_shapes = FlattenArrays({
            k: np.shape(m) for k, m in
            self.resid_means.items()})

        self.bounds = tuple(np.array(list(zip(*[
            b for k, s in param_shapes.items()
            for bound in param_bounds[k]
            for b in repeat(bound, np.prod(s, dtype=int))]))))

    def __call__(self, arr):
        p0 = self.param_shapes.unflatten(arr)
        log_value, det_vars = self.factor_approx.factor(
            {**p0, **self.fixed_kws}
        )
        vals = {**p0, **det_vars}
        residuals = {
            v: (vals[v] - mean) / self.resid_scales[v]
            for v, mean in self.resid_means.items()
        }
        return self.resid_shapes.flatten(residuals)

    def least_squares(self):
        p0 = {
            v: self.factor_approx.model_dist[v].sample(1)[0]
            for v in self.param_shapes.keys()}
        arr = self.param_shapes.flatten(p0)

        res = least_squares(
            self, arr, bounds=self.bounds, **self.opt_params)

        sol = self.param_shapes.unflatten(res.x)
        _, det_vars = self.factor_approx.factor(
            {**sol, **self.fixed_kws}
        )

        jac = {
            (d, k): b
            for k, a in self.param_shapes.unflatten(
                res.jac.T, ndim=1).items()
            for d, b in self.resid_shapes.unflatten(
                a.T, ndim=1).items()}
        hess = self.param_shapes.unflatten(
            res.jac.T.dot(res.jac))

        def inv(a):
            shape = np.shape(a)
            ndim = len(shape)
            if ndim:
                a = np.asanyarray(a)
                s = shape[:ndim // 2]
                n = np.prod(s, dtype=int)
                return np.linalg.inv(
                    a.reshape(n, n)).reshape(s + s)
            else:
                return 1 / a

        invhess = {
            k: inv(h) for k, h in hess.items()}
        for det in det_vars:
            invhess[det] = 0.
            for v in sol:
                invhess[det] += propagate_uncertainty(
                    invhess[v], jac[det, v])

        mode = {**sol, **det_vars}
        return mode, invhess, res


def lstsq_laplace_factor_approx(
        model_approx: MeanFieldApproximation,
        factor: Factor,
        delta: float = 0.5,
        opt_kws: Optional[Dict[str, Any]] = None):
    """
    """
    factor_approx = model_approx.factor_approximation(factor)

    opt = LeastSquaresOpt(
        factor_approx, **({} if opt_kws is None else opt_kws))

    mode, covar, result = opt.least_squares()
    message = (
        "optimise.lsq_sq_laplace_factor_approx: "
        f"nfev={result.nfev}, njev={result.njev}, "
        f"optimality={result.optimality}, "
        f"cost={result.cost}, "
        f"status={result.status}, message={result.message}",)
    status = Status(result.success, message)

    model_dist = {
        v: factor_approx.factor_dist[v].from_mode(
            mode[v],
            covar.get(v))
        for v in mode
    }

    projection, status = factor_approx.project(
        model_dist, delta=delta, status=status)

    return model_approx.project(projection, status=status)
