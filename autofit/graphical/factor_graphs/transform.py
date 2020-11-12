

from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, get_blas_funcs

from autofit.graphical.factor_graphs import \
    AbstractNode, Variable, FactorValue, JacobianValue, HessianValue
from autofit.graphical.utils import cached_property, Axis

class AbstractTransform(ABC):

    @property 
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def whiten(self, values: Dict[Variable, np.ndarray]):
        pass
    
    @abstractmethod
    def unwhiten(self, values: Dict[Variable, np.ndarray]):
        pass

    @abstractmethod
    def whiten2d(self, values: Dict[Variable, np.ndarray]):
        pass
    
    @abstractmethod
    def unwhiten2d(self, values: Dict[Variable, np.ndarray]):
        pass

    @property
    @abstractmethod
    def log_det(self):
        pass

    @cached_property
    def det(self):
        return np.exp(self.log_det)


class RescaleTransform(AbstractTransform):
    def __init__(self, variables_scales: Dict[Variable, np.ndarray]):
        self.scale = variables_scales
        self.inv_scale = {v: scale**-1 for v, scale in self.scale.items()}

    def whiten(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: val * self.inv_scale[v] for v, val in values.items()}

    def unwhiten(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: val * self.scale[v] for v, val in values.items()}

    def whiten2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        whitened = {}
        for v, hess in values.items():
            inv_scale = self.inv_scale[v]
            shape = np.shape(inv_scale)
            size = np.size(inv_scale)
            hess2d = np.reshape(hess, (size, size))
            inv_scale1d = inv_scale.ravel()
            w_hess = hess2d * inv_scale1d[None, :] * inv_scale1d[:, None]
            whitened[v] = w_hess.reshape(shape + shape)

        return whitened

    def unwhiten2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        unwhitened = {}
        for v, hess in values.items():
            scale = self.scale[v]
            shape = np.shape(scale)
            size = np.size(scale)
            hess2d = np.reshape(hess, (size, size))
            scale1d = scale.ravel()
            w_hess = hess2d * scale1d[None, :] * scale1d[:, None]
            unwhitened[v] = w_hess.reshape(shape + shape)

        return unwhitened

    @cached_property
    def log_det(self):
        return sum(np.log(scale).sum() for scale in self.scale.values())

def _solve_triangular(c_and_lower, b, trans=False, overwrite_b=False):
    c, lower = c_and_lower
    return solve_triangular(
        c, b, lower=lower, trans=trans, overwrite_b=overwrite_b)

def _mul_triangular(c_and_lower, b, trans=False, overwrite_b=False):
    c, lower = c_and_lower 
    a1 = np.asarray(c)
    b1 = np.asarray(b)

    trmv, = get_blas_funcs(('trmv',), (a1, b1))
    if a1.flags.f_contiguous:
        return trmv(
            a1, b1, lower=lower, trans=trans, overwrite_x=overwrite_b)
    else:
        # transposed system is solved since trmv expects Fortran ordering
        return trmv(
            a1.T, b1, lower=not lower, trans=not trans, 
            overwrite_x=overwrite_b)

def _whiten_choleksy(c_and_lower, b):
    c, lower = c_and_lower
    b = np.asarray(b)
    n = c.shape[1]
        
    return _solve_triangular(
        c_and_lower, b.reshape(n, -1), trans=lower, 
    ).reshape(b.shape)
        

def _unwhiten_cholesky(c_and_lower, b):
    c, lower = c_and_lower
    b = np.asarray(b)
    n = c.shape[1]


    if b.size == n:
        return _mul_triangular(
            c_and_lower, b.ravel(), trans=lower
        ).reshape(b.shape)
    else:
        d = np.array(b.reshape(n, -1), order='F')
        for i in range(d.shape[1]):
            _mul_triangular(
                c_and_lower, d[:, i], trans=lower,
                overwrite_b = True
            )

        return d.reshape(b.shape)

class CholeskyTransform(AbstractTransform):

    def __init__(
            self,
            variable_cho_factors: Dict[Variable, Tuple[np.ndarray, bool]],
            _inv_transform = None,
            _transform = None,
    ):
        self.cho_factors = variable_cho_factors
        
        self._inv_transform = (
            _unwhiten_cholesky if _inv_transform is None 
            else _inv_transform)
        self._transform = (
            _whiten_choleksy if _transform is None 
            else _transform)

    def whiten(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._transform(self.cho_factors[v], val) 
            for v, val in values.items()}

    def unwhiten(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._inv_transform(self.cho_factors[v], val) 
            for v, val in values.items()}

    def whiten2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._inv_transform(self.cho_factors[v],
                self._inv_transform(self.cho_factors[v], val).T).T
            for v, val in values.items()}

    def unwhiten2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._transform(self.cho_factors[v],
                self._transform(self.cho_factors[v], val).T).T
            for v, val in values.items()}

    @cached_property
    def log_det(self):
        return sum(
            # determinant of triangular matrix is product of diagonal
            np.log(c.diagonal()).sum() 
            for c, _ in self.cho_factors.values()
        )


class TransformedNode(AbstractNode):
    def __init__(
        self, 
        node: AbstractNode, 
        transform: AbstractTransform
    ):
        self.node = node 
        self.transform = transform 

    @property 
    def variables(self):
        self.node.variables 

    @property
    def deterministic_variables(self):
        return self.node.deterministic_variables

    @property
    def all_variables(self):
        return self.node.all_variables

    @property
    def name(self):
        return f"FactorApproximation({self.node.name})"

    def __call__(
            self, 
            values: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        unwhitened = self.transform.whiten(values)
        return self.node(unwhitened, axis=axis)

    def func_jacobian(
            self, 
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
    ) -> Tuple[FactorValue, JacobianValue]:
        unwhittened = self.transform.unwhiten(values)
        fval, jval = self.func_jacobian(
            unwhittened, 
            variables=variables, 
            axis=axis,
            _calc_deterministic=_calc_deterministic)

        jval = self.transform.whiten(jval)
        return fval, jval

    def func_jacobian_hessian(
            self, 
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
    ) -> Tuple[FactorValue, JacobianValue, HessianValue]:
        unwhitened = self.transform.unwhiten(values)
        fval, jval, hval = self.func_jacobian_hessian(
            unwhitened, 
            variables=variables, 
            axis=axis,
            _calc_deterministic=_calc_deterministic)

        jval = self.transform.whiten(jval)
        hval = self.transform.whiten2d(hval)
        return fval, jval, hval
