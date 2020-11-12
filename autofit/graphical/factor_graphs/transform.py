

from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, get_blas_funcs

from autofit.graphical.factor_graphs import \
    AbstractNode, Variable, FactorValue, JacobianValue, HessianValue
from autofit.graphical.utils import cached_property, Axis, FlattenArrays

class AbstractTransform(ABC):

    @property 
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def transform(self, values: Dict[Variable, np.ndarray]):
        pass
    
    @abstractmethod
    def untransform(self, values: Dict[Variable, np.ndarray]):
        pass

    @abstractmethod
    def transform2d(self, values: Dict[Variable, np.ndarray]):
        pass
    
    @abstractmethod
    def untransform2d(self, values: Dict[Variable, np.ndarray]):
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

    @property
    def variables(self):
        return self.scale.keys()

    def transform(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: val * self.inv_scale[v] for v, val in values.items()}

    def untransform(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: val * self.scale[v] for v, val in values.items()}

    def transform2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        transformed = {}
        for v, hess in values.items():
            inv_scale = self.inv_scale[v]
            shape = np.shape(inv_scale)
            size = np.size(inv_scale)
            hess2d = np.reshape(hess, (size, size))
            inv_scale1d = inv_scale.ravel()
            w_hess = hess2d * inv_scale1d[None, :] * inv_scale1d[:, None]
            transformed[v] = w_hess.reshape(shape + shape)

        return transformed

    def untransform2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        actual_values = {}
        for v, hess in values.items():
            scale = self.scale[v]
            shape = np.shape(scale)
            size = np.size(scale)
            hess2d = np.reshape(hess, (size, size))
            scale1d = scale.ravel()
            w_hess = hess2d * scale1d[None, :] * scale1d[:, None]
            actual_values[v] = w_hess.reshape(shape + shape)

        return actual_values

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

def _whiten_cholesky(c_and_lower, b):
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
        # make a copy of b in Fortran memory order
        d = np.array(b.reshape(n, -1), order='F')
        for i in range(d.shape[1]):
            # save result of multiplication in d
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
            _unwhiten_cholesky if _inv_transform is None else _inv_transform)
        self._transform = (
            _whiten_cholesky if _transform is None else _transform)

    def transform(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._transform(self.cho_factors[v], val) 
            for v, val in values.items()}

    def untransform(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._inv_transform(self.cho_factors[v], val) 
            for v, val in values.items()}

    def transform2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        return {
            v: self._inv_transform(self.cho_factors[v],
                self._inv_transform(self.cho_factors[v], val).T).T
            for v, val in values.items()}

    def untransform2d(
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

class FullCholeskyTransform(AbstractTransform):
    def __init__(
            self, 
            cho_factor: Tuple[np.ndarray, bool], 
            param_shapes: FlattenArrays):
        self.c, self.lower = self.cho_factor = cho_factor
        self.param_shapes = param_shapes

    @property
    def variables(self):
        self.param_shapes.keys()       

    def transform(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        x0 = self.param_shapes.flatten(values)
        x1 = _whiten_cholesky(self.cho_factor, x0)
        return self.param_shapes.unflatten(x1)

    def untransform(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        x0 = self.param_shapes.flatten(values)
        x1 = _unwhiten_cholesky(self.cho_factor, x0)
        return self.param_shapes.unflatten(x1)

    def transform2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        X0 = self.param_shapes.flatten2D(values)
        X1 = _whiten_cholesky(
            self.cho_factor,
            _whiten_cholesky(
                self.cho_factor, 
                X0).T
        ).T
        return self.param_shapes.unflatten2d(X1)

    def untransform2d(
        self, 
        values: Dict[Variable, np.ndarray]
    ) -> Dict[Variable, np.ndarray]:
        X0 = self.param_shapes.flatten2D(values)
        X1 = _unwhiten_cholesky(
            self.cho_factor,
            _unwhiten_cholesky(
                self.cho_factor, 
                X0).T
        ).T
        return self.param_shapes.unflatten2d(X1)

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
        actual_values = self.transform.transform(values)
        return self.node(actual_values, axis=axis)

    def func_jacobian(
            self, 
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
    ) -> Tuple[FactorValue, JacobianValue]:
        unwhittened = self.transform.untransform(values)
        fval, jval = self.func_jacobian(
            unwhittened, 
            variables=variables, 
            axis=axis,
            _calc_deterministic=_calc_deterministic)

        jval = self.transform.transform(jval)
        return fval, jval

    def func_jacobian_hessian(
            self, 
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
    ) -> Tuple[FactorValue, JacobianValue, HessianValue]:
        actual_values = self.transform.untransform(values)
        fval, jval, hval = self.func_jacobian_hessian(
            actual_values, 
            variables=variables, 
            axis=axis,
            _calc_deterministic=_calc_deterministic)

        jval = self.transform.transform(jval)
        hval = self.transform.transform2d(hval)
        return fval, jval, hval
