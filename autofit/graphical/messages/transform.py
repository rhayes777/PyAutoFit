

from abc import ABC, abstractmethod
from functools import cached_property, wraps
from typing import Type, Union, Tuple

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats._continuous_distns import _norm_pdf

from ..factor_graphs.transform import (
    DiagonalTransform,
    AbstractArray1DarTransform,
    MatrixOperator
)
from ..factor_graphs import transform


class AbstractTransform(ABC):
    """
    Apply transforms and inverse transforms, 

    def f(x):
        return 2*x
    """
    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inv_transform(self, x):
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> AbstractArray1DarTransform:
        pass

    @abstractmethod
    def log_det(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def log_det_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def transform_det(self, x) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(x), self.log_det(x)

    def transform_jac(self, x) -> Tuple[np.ndarray, AbstractArray1DarTransform]:
        return self.transform(x), self.jacobian(x)

    def transform_det_jac(
        self, x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AbstractArray1DarTransform]:
        return (
            self.transform(x),
            *self.log_det_grad(x),
            self.jacobian(x)
        )

    def transform_func(self, func):
        @wraps(func)
        def transformed_func(*args, **kwargs):
            x, *args = args
            x = self.transform(x)
            return func(x, *args, **kwargs)

        transformed_func.transform = self

        return transformed_func

    def transform_func_grad(self, func_grad):
        @wraps(func_grad)
        def transformed_func_grad(*args, **kwargs):
            x, *args = args
            x, jac = self.transform_jac(x)
            val, grad = func_grad(x, *args, **kwargs)
            return x, grad * jac

        transformed_func_grad.transform = self
        return transformed_func_grad

    def transform_func_grad_hess(self, func_grad_hess):
        @wraps(func_grad_hess)
        def transformed_func_grad_hess(*args, **kwargs):
            x, *args = args
            x, jac = self.transform_jac(x)
            val, grad, hess = func_grad_hess(x, *args, **kwargs)
            return val, grad * jac, jac.quad(hess)

        transformed_func_grad_hess.transform = self
        return transformed_func_grad_hess


class LinearTransform(AbstractTransform):
    def __init__(self, linear: AbstractArray1DarTransform):
        self.linear = linear

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.linear * x

    def inv_transform(self, x: np.ndarray) -> np.ndarray:
        return self.linear.ldiv(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.linear

    def log_det(self, x: np.ndarray) -> np.ndarray:
        return self.linear.log_det

    def log_det_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.log_det(x), 0


class LinearShiftTransform(LinearTransform):
    def __init__(self, shift: float = 0, scale: float = 1):
        self.shift = shift
        self.scale = scale
        self.linear = DiagonalTransform(np.reciprocal(self.scale))

    def inv_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.shift

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.shift) / self.scale

    def log_det(self, x: np.ndarray) -> np.ndarray:
        return - np.log(self.scale) * np.ones_like(x)


class FunctionTransform(AbstractTransform):
    def __init__(self, func, inv_func, grad, hess=None, args=(), func_grad_hess=None):
        self.func = func
        self.inv_func = inv_func
        self.grad = grad
        self.hess = hess
        self.args = ()
        self.func_grad_hess = func_grad_hess

    def transform(self, x):
        return self.func(x, *self.args)

    def inv_transform(self, x):
        return self.inv_func(x, *self.args)

    def jacobian(self, x):
        return DiagonalTransform(self.grad(x, *self.args))

    def log_det(self, x: np.ndarray) -> np.ndarray:
        gs = self.grad(x, *self.args)
        return np.log(gs)

    def log_det_grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.func_grad_hess:
            x0, gs, hs = self.func_grad_hess(x, *self.args)
        else:
            x0 = self.func(x, *self.args)
            gs = self.grad(x, *self.args)
            hs = self.hess(x, *self.args)
        return np.log(gs), np.reciprocal(hs)

    def transform_det_jac(
        self, x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AbstractArray1DarTransform]:
        if self.func_grad_hess:
            x0, gs, hs = self.func_grad_hess(x, *self.args)
        else:
            x0 = self.func(x, *self.args)
            gs = self.grad(x, *self.args)
            hs = self.hess(x, *self.args)
        return x0, np.log(gs), np.reciprocal(hs), DiagonalTransform(gs)


exp_transform = FunctionTransform(np.exp, np.log, np.exp, np.exp)


def reciprocal2(x):
    return np.reciprocal(np.reciprocal(x))


log_transform = FunctionTransform(np.log, np.exp, np.reciprocal, reciprocal2)


def sigmoid(x, scale=1, shift=0):
    return scale / (1 + np.exp(-x)) + shift


def logit(p, scale=1, shift=0):
    p = (p + shift) / scale
    return np.log(p) - np.log1p(-p)


def sigmoid_grad(x, scale=1, shift=0):
    expx = np.exp(-x)
    return scale * expx / np.square(1 + expx)


def logit_grad(x, scale=1, shift=0):
    x = (x - shift) / scale
    return np.reciprocal(x) + np.reciprocal(1 - x)


def logit_hess(x, scale=1, shift=0):
    x = (x - shift) / scale
    return np.reciprocal2(1 - x) - np.reciprocal2(x)


logistic_transform = FunctionTransform(logit, sigmoid, logit_grad, logit_hess)


def ndtri_grad(x):
    return np.reciprocal(_norm_pdf(ndtri(x)))


def ndtri_hess(x):
    return


def ndtri_grad_hess(x):
    f = ndtri(x)
    phi = _norm_pdf(f)
    grad = np.reciprocal(phi)
    hess = grad**2 * f
    return f, grad, hess


phi_transform = FunctionTransform(
    ndtri, ndtr, ndtri_grad, func_grad_hess=ndtri_grad_hess)


class MultinomialLogitTransform(AbstractTransform):
    def transform(self, p):
        kws = {'axis': -1, 'keepdims': True} if np.ndim(p) > 1 else {}
        lnp1 = np.log(1 - np.sum(p, **kws))
        lnp = np.log(p)
        return lnp - lnp1

    def inv_transform(self, x):
        kws = {'axis': -1, 'keepdims': True} if np.ndim(x) > 1 else {}
        expx = np.exp(x)
        return expx / (expx.sum(**kws) + 1)

    def jacobian(self, p):
        # TODO define custom linear transform
        # to take advantage of Sherman Morrison
        if np.ndim(p) > 1:
            raise NotImplementedError(
                'jacobian for mutiple samples not defined')
        n = np.shape(p)[-1]
        pn1 = 1 - np.sum(p, axis=-1)
        M = np.full((n, n), 1/pn1)
        M.flat[::n+1] += 1/p
        return MatrixOperator(M)

    def log_det(self, p):
        kws = {'axis': -1, 'keepdims': True} if np.ndim(p) > 1 else {}
        p1 = 1 - np.sum(p, **kws)
        log_d = - np.log(p).sum(**kws) - np.log(p1)
        # Hack to make sure summation broadcasting works correctly
        return log_d * np.full_like(p, p1.size / p.size)

    def log_det_grad(self, p):
        kws = {'axis': -1, 'keepdims': True} if np.ndim(p) > 1 else {}
        p1 = 1 - np.sum(p, **kws)
        log_d = - np.log(p).sum(**kws) - np.log(p1)
        return log_d * np.full_like(p, p1.size / p.size), 1/p1 - 1/p

    def transform_det_jac(
        self, p
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AbstractArray1DarTransform]:
        p = np.asanyarray(p)
        if np.ndim(p) > 1:
            raise NotImplementedError(
                'jacobian for mutiple samples not defined')

        n = np.shape(p)[-1]
        pn1 = 1 - np.sum(p)
        ln1p = np.log(pn1)
        lnp = np.log(p)
        x = lnp - ln1p
        logd = np.full_like(p,  (- lnp.sum() - ln1p) / p.size)
        logd_grad = 1/pn1 - 1/p
        M = np.full((n, n), 1/pn1)
        M.flat[::n+1] += 1/p
        jac = MatrixOperator(M)
        return (
            x, logd, logd_grad, jac
        )
