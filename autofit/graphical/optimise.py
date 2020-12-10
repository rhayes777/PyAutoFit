from collections import defaultdict
from itertools import repeat
from typing import (
    Optional, Dict, Tuple, Any, List, Iterator
)

import numpy as np
from scipy.optimize import (
    minimize,
    OptimizeResult,
    least_squares,
    approx_fprime
)

from autofit.graphical.expectation_propagation import (
    EPMeanField,
    AbstractFactorOptimiser
)
from autofit.graphical.factor_graphs import (
    Variable,
    Factor,
    JacobianValue
)
from autofit.graphical.factor_graphs.transform import (
    AbstractLinearTransform,
    identity_transform,
    CovarianceTransform
)
from autofit.graphical.mean_field import (
    MeanField,
    FactorApproximation,
    Status
)
from autofit.graphical.messages import FixedMessage
from autofit.graphical.utils import (
    propagate_uncertainty,
    FlattenArrays,
    OptResult
)


class OptFactor:
    """
    """

    def __init__(
            self,
            factor: Factor,
            param_shapes: FlattenArrays,
            fixed_kws: Optional[Dict[str, np.ndarray]] = None,
            model_dist: Optional[MeanField] = None,
            transform: Optional[AbstractLinearTransform] = None,
            bounds: Optional[Dict[str, Tuple[float, float]]] = None,
            method: str = 'L-BFGS-B',
    ):
        self.factor = factor
        self.param_shapes = param_shapes
        self._model_dist = model_dist

        self.transform = identity_transform if transform is None else transform
        self.param_bounds = bounds
        self.free_vars = tuple(self.param_shapes.keys())
        self.deterministic_variables = self.factor.deterministic_variables

        self.sign = 1
        self.fixed_kws = fixed_kws

        meth = method.lower()
        # method needs to return Hessian information.
        if meth not in ('bfgs', 'l-bfgs-b'):
            raise ValueError('Unknown solver %s' % method)

        self.method = meth

        if bounds:
            # TODO check that this is correct for composite
            # distributions e.g. NormalGammaMessage
            self.bounds = [
                b for k, s in self.param_shapes.items()
                for bound in bounds[k]
                for b in repeat(bound, np.prod(s, dtype=int))]
        else:
            self.bounds = bounds

    @property
    def model_dist(self):
        if self._model_dist is None:
            raise ValueError("Model dist not defined")
        else:
            return self._model_dist

    @classmethod
    def from_approx(
            cls,
            factor_approx: FactorApproximation,
            transform: Optional[AbstractLinearTransform] = None,
    ) -> 'OptFactor':
        value_shapes = {}
        fixed_kws = {}
        bounds = {}
        for v in factor_approx.variables:
            dist = factor_approx.model_dist[v]
            if isinstance(dist, FixedMessage):
                fixed_kws[v] = dist.mean
            else:
                value_shapes[v] = dist.shape
                bounds[v] = dist._support

        return cls(
            factor_approx,
            FlattenArrays(value_shapes),
            fixed_kws=fixed_kws,
            model_dist=factor_approx.model_dist,
            transform=transform,
            bounds=bounds,
        )

    def flatten(self, values: Dict[Variable, np.ndarray]) -> np.ndarray:
        x0 = self.param_shapes.flatten(values)
        return x0

    def unflatten(self, x0: np.ndarray) -> Dict[Variable, np.ndarray]:
        values = {**self.param_shapes.unflatten(x0), **self.fixed_kws}
        return values

    def __call__(self, x0):
        values = self.unflatten(self.transform.ldiv(x0))
        return self.sign * np.sum(self.factor(values, axis=None))

    def func_jacobian(self, x0):
        values = self.unflatten(self.transform.ldiv(x0))
        fval, jval = self.factor.func_jacobian(
            values, self.free_vars,
            axis=None, _calc_deterministic=True)

        grad = self.flatten(jval) / self.transform
        return self.sign * fval.log_value, self.sign * grad

    def jacobian(self, args):
        return self.func_jacobian(args)[1]

    def numerically_verify_jacobian(
            self,
            n_tries=10,
            eps=1e-6,
            rtol=1e-3,
            atol=1e-2):
        x0s = (
            self.flatten(self.get_random_start())
            for _ in range(n_tries)
        )
        return all(
            np.allclose(
                self.jacobian(x0),
                approx_fprime(x0, self, eps),
                atol=atol,
                rtol=rtol
            )
            for x0 in x0s
        )

    def get_random_start(self, arrays_dict: Dict[Variable, np.ndarray] = {}):
        values = {
            v: arrays_dict[v] if v in arrays_dict
            else self.model_dist[v].sample()
            for v in self.free_vars
        }
        # transform values
        return self.unflatten(
            self.transform.ldiv(
                self.flatten(values)
            )
        )

    def _parse_result(
            self,
            result: OptimizeResult,
            status: Status = Status()) -> OptResult:
        success, messages = status
        success = result.success
        message = result.message.decode()
        messages += (
            "optimise.find_factor_mode: "
            f"nfev={result.nfev}, nit={result.nit}, "
            f"status={result.status}, message={message}",)

        full_hess_inv = result.hess_inv
        if not isinstance(full_hess_inv, np.ndarray):
            # if optimiser is L-BFGS-B then convert
            # implicit hess_inv into dense matrix
            full_hess_inv = full_hess_inv.todense()

        # make inverse transform back
        M = self.transform
        x = M.ldiv(result.x)
        full_hess_inv = M.ldiv(M.ldiv(full_hess_inv).T)

        mode = {**self.param_shapes.unflatten(x), **self.fixed_kws}
        hess_inv = self.param_shapes.unflatten(full_hess_inv)

        return OptResult(
            mode,
            hess_inv,
            self.sign * result.fun,  # minimized negative logpdf of factor approximation
            full_hess_inv,  # full inverse hessian of optimisation
            result,
            Status(success, messages))

    def _minimise(self, arrays_dict, method=None, bounds=None,
                  constraints=(), tol=None, callback=None,
                  options=None):
        x0 = self.transform * self.param_shapes.flatten(arrays_dict)
        bounds = self.bounds if bounds is None else bounds
        method = self.method if method is None else method
        return minimize(
            self.func_jacobian, x0, method=method, jac=True, bounds=bounds,
            constraints=constraints, tol=tol, callback=callback,
            options=options)

    def minimise(
            self,
            arrays_dict: Dict[Variable, np.ndarray] = {},
            bounds=None,
            constraints=(),
            tol=None,
            callback=None,
            options=None,
            status: Status = Status(),
    ):
        self.sign = 1
        p0 = self.get_random_start(arrays_dict)
        res = self._minimise(
            p0,
            bounds=bounds, constraints=constraints, tol=tol,
            callback=callback, options=options)
        return self._parse_result(res, status=status)

    def maximise(
            self,
            arrays_dict: Dict[Variable, np.ndarray] = {},
            bounds=None,
            constraints=(),
            tol=None,
            callback=None,
            options=None,
            status: Status = Status(),
    ):
        self.sign = -1
        p0 = self.get_random_start(arrays_dict)
        res = self._minimise(
            p0,
            bounds=bounds, constraints=constraints, tol=tol,
            callback=callback, options=options)
        return self._parse_result(res, status=status)

    minimize = minimise
    maximize = maximise


def update_det_cov(
        res: OptResult,
        jacobian: JacobianValue):
    """Calculates the inv hessian of the deterministic variables

    Note that this modifies res.
    """
    covars = res.hess_inv
    for v, grad in jacobian.items():
        for det, jac in grad.items():
            cov = propagate_uncertainty(covars[v], jac)
            covars[det] = covars.get(det, 0.) + cov

    return res


class LaplaceFactorOptimiser(AbstractFactorOptimiser):

    def __init__(
            self,
            whiten_optimiser=True,
            transforms=None,
            deltas=None,
            opt_kws=None):

        self.whiten_optimiser = whiten_optimiser
        self.transforms = defaultdict(lambda: identity_transform)
        if transforms:
            self.transforms.update(transforms)

        self.deltas = defaultdict(lambda: 1)
        if deltas:
            self.deltas.update(deltas)

        self.opt_kws = defaultdict(dict)
        if opt_kws:
            self.opt_kws.update(opt_kws)

    def optimise(
            self,
            factor: Factor,
            model_approx: EPMeanField,
            status: Optional[Status] = Status(),
    ) -> Tuple[EPMeanField, Status]:

        whiten = self.transforms[factor]
        delta = self.deltas[factor]
        opt_kws = self.opt_kws[factor]

        factor_approx = model_approx.factor_approximation(factor)
        opt = OptFactor.from_approx(factor_approx, transform=whiten)
        res = opt.maximise(status=status, **opt_kws)

        # Calculate covariance of deterministic values
        # TODO: estimate this Jacobian using Broyden's method
        # https://en.wikipedia.org/wiki/Broyden%27s_method
        value = factor_approx.factor(res.mode)
        res.mode.update(value.deterministic_values)
        jacobian = factor_approx.factor.jacobian(
            res.mode, opt.free_vars, axis=None)
        update_det_cov(res, jacobian)

        self.transforms[factor] = CovarianceTransform.from_dense(
            res.full_hess_inv)

        # Project Laplace's approximation
        new_model_dist = factor_approx.model_dist.project_mode(res)
        projection, status = factor_approx.project(
            new_model_dist,
            delta=delta,
            status=res.status
        )
        new_approx, status = model_approx.project(projection, status)
        return new_approx, status


LaplaceFactorOptimizer = LaplaceFactorOptimiser


#################################################


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
        status: Status = Status(),
        min_iter: int = 2,
        opt_kws: Optional[dict] = None,
        **kwargs
) -> OptResult:
    """
    """
    opt_kws = {} if opt_kws is None else opt_kws

    opt = OptFactor.from_approx(factor_approx, **kwargs)
    res = opt.maximise(status=status, **opt_kws)

    if return_cov:
        # Calculate deterministic values
        value = factor_approx.factor(res.mode)
        res.mode.update(value.deterministic_values)

        # Calculate covariance of deterministic values
        jacobian = factor_approx.factor.jacobian(
            res.mode, opt.free_vars)
        update_det_cov(res, jacobian)

    return res


def laplace_factor_approx(
        model_approx: EPMeanField,
        factor: Factor,
        delta: float = 1.,
        status: Status = Status(),
        opt_kws: Optional[Dict[str, Any]] = None
):
    opt_kws = {} if opt_kws is None else opt_kws
    factor_approx = model_approx.factor_approximation(factor)
    res = find_factor_mode(
        factor_approx,
        return_cov=True,
        status=status,
        **opt_kws
    )

    model_dist = factor_approx.model_dist.project_mode(res)
    projection, status = factor_approx.project(
        model_dist,
        delta=delta,
        status=res.status
    )

    new_approx, status = model_approx.project(
        projection, status=status)

    return new_approx, status


class LaplaceOptimiser:
    def __init__(
            self,
            n_iter=4,
            delta=1.,
            opt_kws: Optional[Dict[str, Any]] = None
    ):
        self.history = dict()
        self.n_iter = n_iter
        self.delta = delta
        self.opt_kws = {} if opt_kws is None else opt_kws

    def step(
            self,
            model_approx,
            factors: Optional[List[Factor]] = None,
            status: Status = Status()
    ) -> Iterator[Tuple[Factor, EPMeanField, Status]]:
        new_approx = model_approx
        factors = (
            model_approx.factor_graph.factors
            if factors is None else factors)
        for factor in factors:
            new_approx, status = laplace_factor_approx(
                new_approx,
                factor,
                self.delta,
                status=status,
                opt_kws=self.opt_kws)
            yield factor, new_approx, status

    def run(
            self,
            model_approx: EPMeanField,
            factors: Optional[List[Factor]] = None,
            status: Status = Status()
    ) -> EPMeanField:
        new_approx = model_approx
        for i in range(self.n_iter):
            for factor, new_approx, status in self.step(new_approx, factors):
                self.history[i, factor] = new_approx
        return new_approx, status


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
                v for v, d in factor_approx.cavity_dist.items()
                if not isinstance(d, FixedMessage)
            )

        self.opt_only = opt_only
        self.resid_means = {
            k: factor_approx.cavity_dist[k].mean for k in self.opt_only}
        self.resid_scales = {
            k: factor_approx.cavity_dist[k].scale for k in self.opt_only}
        self.resid_shapes = FlattenArrays({
            k: np.shape(m) for k, m in self.resid_means.items()})

        self.bounds = tuple(np.array(list(zip(*[
            b for k, s in param_shapes.items()
            for bound in param_bounds[k]
            for b in repeat(bound, np.prod(s, dtype=int))]))))

    def __call__(self, arr):
        p0 = self.param_shapes.unflatten(arr)
        values = {**p0, **self.fixed_kws}
        fvals = self.factor_approx.factor(values)
        values.update(fvals.deterministic_values)
        residuals = {
            v: (values[v] - mean) / self.resid_scales[v]
            for v, mean in self.resid_means.items()
        }
        return self.resid_shapes.flatten(residuals)

    def least_squares(self, values={}):
        model_dist = self.factor_approx.model_dist
        p0 = {
            v: values[v] if v in values else model_dist[v].sample()
            for v in self.param_shapes.keys()}
        arr = self.param_shapes.flatten(p0)

        res = least_squares(
            self, arr, bounds=self.bounds, **self.opt_params)

        sol = self.param_shapes.unflatten(res.x)
        fval = self.factor_approx.factor(
            {**sol, **self.fixed_kws}
        )
        det_vars = fval.deterministic_values

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
        model_approx: EPMeanField,
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

    model_dist = MeanField({
        v: factor_approx.factor_dist[v].from_mode(
            mode[v],
            covar.get(v))
        for v in mode
    })

    projection, status = factor_approx.project(
        model_dist, delta=delta, status=status)

    return model_approx.project(projection, status=status)
