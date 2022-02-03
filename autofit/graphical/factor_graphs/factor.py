from abc import ABC
from functools import lru_cache
from inspect import getfullargspec
from itertools import chain, repeat
from typing import Tuple, Dict, Union, Any, Callable, List, Optional

import numpy as np


try:
    import jax

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from autoconf import cached_property


from autofit.graphical.utils import (
    nested_filter,
    nested_update,
    is_variable,
    aggregate,
    Axis,
)
from autofit.mapper.variable import Variable, Plate, VariableData, broadcast_plates

from autofit.graphical.factor_graphs.jacobians import (
    AbstractJacobian,
    VectorJacobianProduct,
    JacobianVectorProduct,
)
from autofit.graphical.factor_graphs.abstract import FactorValue, AbstractFactor
from autofit.graphical.factor_graphs.abstract import FactorValue, AbstractFactor

# from autofit.graphical.mean_field import MeanField


class Factor(AbstractFactor):
    """
    Examples
    --------
    def linear(x, a, b):
        z = x.dot(a) + b
        return (z**2).sum(), z

    def likelihood(y, z):
        return ((y - z)**2).sum()

    def combined(x, y, a, b):
        like, z = linear(x, a, b)
        return like + likelihood(y, z)

    x_, a_, b_, y_, z_ = variables("x, a, b, y, z")
    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    values = {x_: x, y_: y, a_: a, b_: b}
    linear_factor = FactorVJP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))
    like_factor = FactorVJP(likelihood, y_, z_)
    full_factor = FactorVJP(combined, x_, y_, a_, b_)

    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    values = {x_: x, y_: y, a_: a, b_: b}

    # Fully working problem
    fval, jac = full_factor.func_jacobian(values)
    grad = jac(1)

    linear_val, linear_jac = linear_factor.func_jacobian(values)
    like_val, like_jac = like_factor.func_jacobian(
        {**values, **linear_val.deterministic_values})
    combined_val = like_val + linear_val

    combined_grads = {FactorValue: 1.}
    for v, g in like_jac(combined_grads).items():
        combined_grads[v] = g + combined_grads.get(v, 0)

    for v, g in linear_jac(combined_grads).items():
        combined_grads[v] = g + combined_grads.get(v, 0)

    assert (fval.log_value - combined_val.log_value) == 0
    assert (grad - combined_grads).norm() == 0
    """

    def __init__(
        self,
        factor,
        *args: Variable,
        name="",
        arg_names=None,
        factor_out=FactorValue,
        plates: Tuple[Plate, ...] = (),
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
        eps=1e-8,
    ):
        if not arg_names:
            arg_names = [arg for arg in getfullargspec(factor).args]
            if arg_names[0] == "self":
                arg_names = arg_names[1:]

            # Make sure arg_names matches length of args
            for v in args[len(arg_names) :]:
                arg_name = v.name
                # Make sure arg_name is unique
                while arg_name in arg_names:
                    arg_name += "_"
                arg_names.append(arg_name)

        # self._args = args
        # self._arg_names = arg_names
        kwargs = dict(zip(arg_names, args))
        name = name or factor.__name__
        super().__init__(name=name, plates=plates, **kwargs)

        self.factor_out = factor_out
        det_variables = set(v[0] for v in nested_filter(is_variable, factor_out))
        det_variables.discard(FactorValue)
        self._deterministic_variables = det_variables

        self.eps = eps
        self._set_factor(factor)
        self._set_jacobians(
            vjp=vjp,
            factor_vjp=factor_vjp,
            factor_jacobian=factor_jacobian,
            jacobian=jacobian,
            numerical_jacobian=numerical_jacobian,
            jacfwd=jacfwd,
        )

    # @property
    # def args(self):
    #     return self._args

    # @property
    # def arg_names(self):
    #     return self._arg_names

    def _set_jacobians(
        self,
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
    ):
        if vjp:
            if factor_vjp:
                self._factor_vjp = factor_vjp
            elif not _HAS_JAX:
                raise ModuleNotFoundError(
                    "jax needed if `factor_vjp` not passed with vjp=True"
                )
            else:
                self._factor_vjp = self._jax_factor_vjp

            self.func_jacobian = self._vjp_func_jacobian
        else:
            # This is set by default
            # self.func_jacobian = self._jvp_func_jacobian

            if factor_jacobian:
                self._factor_jacobian = factor_jacobian
            elif jacobian:
                self._jacobian = jacobian
            elif numerical_jacobian:
                self._factor_jacobian = self._numerical_factor_jacobian
            elif _HAS_JAX:
                if jacfwd:
                    self._jacobian = jax.jacfwd(self._factor, range(self.n_args))
                else:
                    self._jacobian = jax.jacobian(self._factor, range(self.n_args))

    def _factor_value(self, raw_fval):
        """Converts the raw output of the factor into a `FactorValue`
        where the values of the deterministic values are stored in a dict
        attribute `FactorValue.deterministic_values`
        """
        det_values = VariableData(nested_filter(is_variable, self.factor_out, raw_fval))
        fval = det_values.pop(FactorValue, 0.0)
        return FactorValue(fval, det_values)

    def __call__(self, values: VariableData):
        """Calls the factor with the values specified by the dictionary of
        values passed, returns a FactorValue with the value returned by the
        factor, and any deterministic factors"""
        raw_fval = self._factor_args(*(values[v] for v in self.args))
        return self._factor_value(raw_fval)

    def _jax_factor_vjp(self, *args) -> Tuple[Any, Callable]:
        return jax.vjp(self._factor, *args)

    def _vjp_func_jacobian(
        self, values: VariableData
    ) -> Tuple[FactorValue, VectorJacobianProduct]:
        """Calls the factor and returns the factor value with deterministic
        values, and a `VectorJacobianProduct` operator that allows the
        calculation of the gradient of the input values to be calculated
        with respect to the gradients of the output values (i.e backprop)
        """
        raw_fval, fvjp = self._factor_vjp(*(values[v] for v in self.args))
        fval = self._factor_value(raw_fval)

        fvjp_op = VectorJacobianProduct(
            self.factor_out,
            fvjp,
            *self.args,
            out_shapes=fval.to_dict().shapes,
        )
        return fval, fvjp_op

    def _jvp_func_jacobian(
        self, values: VariableData, **kwargs
    ) -> Tuple[FactorValue, JacobianVectorProduct]:
        args = (values[k] for k in self.args)
        raw_fval, raw_jac = self._factor_jacobian(*args, **kwargs)
        fval = self._factor_value(raw_fval)
        jvp = self._jac_out_to_jvp(raw_jac, values=fval.to_dict().merge(values))
        return fval, jvp

    func_jacobian = _jvp_func_jacobian

    def _factor_jacobian(self, *args) -> Tuple[Any, Any]:
        return self._factor_args(*args), self._jacobian(*args)

    def _factor_args(self, *args):
        return self._factor(*args)

    def _unpack_jacobian_out(self, raw_jac: Any) -> Dict[Variable, VariableData]:
        jac = {}
        for v0, vjac in nested_filter(is_variable, self.factor_out, raw_jac):
            jac[v0] = VariableData()
            for v1, j in zip(self.args, vjac):
                jac[v0][v1] = j

        return jac

    def _jac_out_to_jvp(
        self, raw_jac: Any, values: VariableData
    ) -> JacobianVectorProduct:
        jac = self._unpack_jacobian_out(raw_jac)
        return JacobianVectorProduct.from_dense(jac, values=values)


class FactorKW(Factor):
    def __init__(
        self,
        factor,
        name="",
        arg_names=None,
        factor_out=FactorValue,
        plates: Tuple[Plate, ...] = (),
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
        eps=1e-8,
        **kwargs: Variable,
    ):
        args = tuple(kwargs.values())
        arg_names = tuple(kwargs.keys())
        super().__init__(
            factor,
            *args,
            name=name,
            arg_names=arg_names,
            factor_out=factor_out,
            plates=plates,
            vjp=vjp,
            factor_vjp=factor_vjp,
            factor_jacobian=factor_jacobian,
            jacobian=jacobian,
            numerical_jacobian=numerical_jacobian,
            jacfwd=jacfwd,
            eps=eps,
        )

    _factor_args = AbstractFactor._factor_args
