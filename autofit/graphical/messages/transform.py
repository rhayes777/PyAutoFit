

from abc import ABC, abstractmethod
from functools import cached_property, wraps 
from typing import Type, Union

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats._continuous_distns import _norm_pdf

from ..factor_graphs.transform import (
    DiagonalTransform,
    AbstractArray1DarTransform,
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

    def transform_grad(self, x, g):
        return g * self.jacobian(x)  

    def inv_transform_grad(self, x, g):
        return g / self.jacobian(x)

    def inv_transform_cov(self, x, H):
        jac = self.jacobian(x) 
        return (H * jac).T * jac 

    def transform_cov(self, x, H):
        jac = self.jacobian(x) 
        return (H / jac).T / jac 

    transform_hess = inv_transform_cov
    inv_transform_hess = transform_cov

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
            x = self.transform(x)
            val, grad = func_grad(x, *args, **kwargs)
            grad = self.transform_grad(x, grad)
            return x, grad 

        transformed_func_grad.transform = self 
        return transformed_func_grad 

    def transform_func_grad_hess(self, func_grad_hess):
        @wraps(func_grad_hess)
        def transformed_func_grad_hess(*args, **kwargs):
            x, *args = args
            x = self.transform(x)
            val, grad, hess = func_grad_hess(x, *args, **kwargs)
            grad = self.inv_transform_grad(x, grad)
            hess = self.inv_transform_hess(x, hess)
            return val, grad, hess

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


class LinearShiftTransform(LinearTransform):
    def __init__(self, shift: float = 0, scale: float = 1):
        self.shift = shift 
        self.scale = scale
        self.linear = DiagonalTransform(self.scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x  * self.scale  + self.shift

    def inv_transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.shift) / self.scale 

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.linear


class FunctionTransform(AbstractTransform):
    def __init__(self, func, inv_func, grad, args=()):
        self.func = func 
        self.inv_func = inv_func 
        self.grad = grad
        self.args = ()

    def transform(self, x):
        return self.func(x, *self.args)

    def inv_transform(self, x):
        return self.inv_func(x, *self.args)

    def jacobian(self, x):
        return DiagonalTransform(self.grad(x, *self.args))

exp_transform = FunctionTransform(np.exp, np.log, np.exp)
log_transform = FunctionTransform(np.log, np.exp, np.reciprocal)

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

logistic_transform = FunctionTransform(sigmoid, logit, logit_grad)

def ndtri_grad(x):
    return np.reciprocal(_norm_pdf(ndtri(x)))

phi_transform = FunctionTransform(ndtr, ndtri, ndtri_grad)

# class TransformedMessage(AbstractMessage):

#     def __new__(cls, message, transform, *args, **kwargs):
#         Transformed = cls.transform_message(message, transform)
#         return object.__new__(Transformed)

#     @classmethod
#     def transform_message(cls, message, transform):
#         if issubclass(message, AbstractMessage):
#             Message = message 
#         else:
#             Message = type(message)

#         support = tuple(zip(*map(
#             transform.transform, map(np.array, zip(*Message._support))
#         )))
#         projectionClass = (
#             None if Message._projection_class is None 
#             else TransformedMessage(Message._projection_class, transform)
#         )
#         clsname = f"Transformed{Message.__name__}"
#         class Transformed(cls, Message):
#             __qualname__ = clsname
#             _Message = Message
#             _transform = transform 
#             _support = support 
#             __projection_class = projectionClass

#             def __init__(self, message, transform, *args, **kwargs):
#                 Message.__init__(self, *args, **kwargs)

#             parameter_names = Message.parameter_names

#         Transformed.__name__ = clsname

#         return Transformed

#     def __reduce__(self):
#         return (
#             TransformedMessage,
#             (
#                 self._Message, 
#                 self._transform,
#                 *self.parameters, 
#                 self.log_norm 
#             ), 
#         )
 
#     @classmethod 
#     def to_canonical_form(cls, x):
#         x = cls._transform.inv_transform(x)
#         return cls.Message.to_canonical_form(x)

#     @cached_property
#     def mean(self):
#         return self._transform.transform(self.Message.mean.func(self))

#     @cached_property
#     def variance(self):
#         return self._transform.transform(self.Message.variance.func(self))
    
#     def sample(self, n_samples=None):
#         x = super().sample(n_samples)
#         return self._transform.transform(x)

#     def logpdf_gradient(self, x):
#         logl, grad = super().logpdf_gradient(x)
#         jac = self._transform.jacobian(x)
#         return logl, jac * grad

#     def logpdf_gradient_hessian(self, x):
#         logl, grad, hess = super().logpdf_gradient(x)
#         jac = self._transform.jacobian(x)
#         return logl, jac * grad, jac.quad(hess)

# def shift_message(
#     message: Union[AbstractMessage, Type[AbstractMessage]], 
#     shift: float = 0, scale: float = 1
# ):
#     shift_transform = transform.LinearShiftTransform(
#         shift=shift, scale=scale 
#     )
#     return TransformedMessage(
#         message, shift_transform
#     )
