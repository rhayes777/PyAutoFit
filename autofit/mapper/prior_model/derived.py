"""
Derived parameters: a model component's own re-parameterisation of some of its
free parameters, for plotting and reporting.

Many components are *sampled* in one parameterisation and *read* in another: a
shear is fit as ``(gamma_1, gamma_2)`` but has a magnitude and an angle, an
ellipse as two elliptical components but has an axis ratio and a position
angle, a multipole as two components but has a strength and a phase. A component
class declares the conversion as a class attribute::

    class ExternalShear:
        derived_parameters = {
            ("gamma_1", "gamma_2"): shear_magnitude_and_angle_dict_from,
        }

keyed by the names of the free parameters it replaces and valued by a function
that takes every parameter of the component by name (the free ones as arrays
over the samples, the fixed ones as floats) and returns a ``dict`` of derived
name to array. The declarations of a class's bases are inherited, so a subclass
adds to them rather than restating them.

``DerivedParameters`` applies those declarations to an array of samples: the
replaced parameters go, the derived ones take their place (at the position of
the first replaced one, in model order), everything else passes through
unchanged, and each derived parameter carries the superscript of the component
it came from. A declaration whose replaced parameters are not all free in this
model is skipped, since there is nothing to convert.

An angle is periodic, and a conversion fixes where its branch cut falls: a
position angle in ``[-45, 135)`` whose posterior peaks near 135 is split in two,
and its corner plot shows a double peak that is not there. A conversion returns
such a value as ``Periodic(angle, period)`` and the samples are recentred on
their (weighted) circular mean, so the cut sits opposite the peak.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from autonerves import conf

from autofit.text import formatter as frm

DerivedFunction = Callable[..., Dict[str, np.ndarray]]


class Periodic:
    def __init__(self, values, period: float):
        """
        A derived parameter that is periodic (an angle), returned by a conversion so
        the samples are recentred on their circular mean; see the module docstring.

        Parameters
        ----------
        values
            The derived values over the samples, in whatever range the conversion gives.
        period
            The period, e.g. 180 for a position angle, ``360 / m`` for a multipole phase.
        """
        self.values = np.asarray(values, dtype=float)
        self.period = float(period)

    def recentred(self, weights=None) -> np.ndarray:
        """
        The values wrapped into the period centred on their weighted circular mean,
        so a posterior that straddled the branch cut is contiguous.
        """
        values = self.values
        if values.size == 0:
            return values
        theta = 2.0 * np.pi * values / self.period
        weights = (
            np.ones_like(values)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        weights = np.broadcast_to(weights, values.shape)
        mean = (
            np.arctan2(np.sum(weights * np.sin(theta)), np.sum(weights * np.cos(theta)))
            * self.period
            / (2.0 * np.pi)
        )
        half = self.period / 2.0
        return (values - mean + half) % self.period + mean - half


def derived_parameters_for_class(cls) -> Dict[Tuple[str, ...], DerivedFunction]:
    """
    The derived-parameter declarations of a class and all of its bases, the most
    derived class last so it can override an entry of a base.
    """
    declarations = {}
    for base in reversed(getattr(cls, "__mro__", (cls,))):
        declarations.update(base.__dict__.get("derived_parameters", {}) or {})
    return declarations


class DerivedParameters:
    def __init__(
        self,
        names: List[str],
        superscripts: List[Optional[str]],
        component_paths: List[Tuple[str, ...]],
        parameter_array: np.ndarray,
        derived_mask: List[bool],
    ):
        """
        Parameters of a model after each component's declared conversions are
        applied to an array of samples; see the module docstring.

        Parameters
        ----------
        names
            The name of every parameter of the derived set, in order.
        superscripts
            The superscript of each, that of the component it belongs to.
        component_paths
            The named-component path of each (see ``AbstractPriorModel.component_paths``).
        parameter_array
            The samples of the derived set, shape ``(n_samples, len(names))``.
        derived_mask
            Whether each parameter was derived (``True``) or passed through.
        """
        self.names = names
        self.superscripts = superscripts
        self.component_paths = component_paths
        self.parameter_array = parameter_array
        self.derived_mask = derived_mask

    @property
    def has_derived(self) -> bool:
        """Whether any declaration was applied, so the set differs from the model's own."""
        return any(self.derived_mask)

    @property
    def parameter_labels(self) -> List[str]:
        return [
            frm.convert_name_to_label(parameter_name=name, name_to_label=True)
            for name in self.names
        ]

    @property
    def parameter_labels_with_superscripts(self) -> List[str]:
        return [
            f"{label}^{{\\rm {superscript}}}" if superscript else label
            for label, superscript in zip(self.parameter_labels, self.superscripts)
        ]

    @property
    def parameter_labels_with_superscripts_latex(self) -> List[str]:
        return [f"${label}$" for label in self.parameter_labels_with_superscripts]

    @property
    def parameter_indices_by_component(self) -> Dict[str, List[int]]:
        """As ``AbstractPriorModel.parameter_indices_by_component``, over the derived set."""
        groups = {}
        for i, path in enumerate(self.component_paths):
            groups.setdefault(path[0] if path else "model", []).append(i)
        return groups


def _flat_fixed_values(owner) -> Dict[str, float]:
    """
    The fixed (non-prior) numeric attributes of a model component by leaf
    parameter name, a fixed tuple ``centre=(0.0, 0.0)`` as ``centre_0``,
    ``centre_1``, so a conversion can take a constant such as a multipole's order.
    """
    values = {}
    for name, value in vars(owner).items():
        if name.startswith("_") or name in ("cls", "id"):
            continue
        if isinstance(value, tuple):
            for i, item in enumerate(value):
                try:
                    values[f"{name}_{i}"] = float(item)
                except (TypeError, ValueError):
                    pass
            continue
        if isinstance(value, bool):
            continue
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            pass
    return values


def derived_parameters_from(
    model, parameter_array: np.ndarray, weights=None
) -> DerivedParameters:
    """
    Apply every component's declared conversions to an array of samples of the
    model; see the module docstring. Called through
    ``AbstractPriorModel.derived_parameters_from``.

    Parameters
    ----------
    weights
        The weight of each sample, used to centre a ``Periodic`` derived parameter
        on its posterior rather than on the prior volume; equal by default.
    """
    parameter_array = np.asarray(parameter_array)
    n_samples = parameter_array.shape[0]

    owners = model._prior_owner_and_leaf_names
    names = model.parameter_names
    superscripts = model.superscripts
    component_paths = model.component_paths

    # Which free parameters each component owns, by leaf name -> model index.
    indices_by_owner = {}
    for i, (owner, leaf) in enumerate(owners):
        indices_by_owner.setdefault(id(owner), (owner, {}))[1][leaf] = i

    # Resolve every applicable declaration first: which indices it consumes and
    # what it yields, keyed by the index it is inserted at.
    insert_at = {}
    consumed = set()
    for owner, free in indices_by_owner.values():
        cls = getattr(owner, "cls", None)
        if cls is None:
            continue
        for replaced, function in derived_parameters_for_class(cls).items():
            if not all(name in free for name in replaced):
                continue
            if any(free[name] in consumed for name in replaced):
                continue
            values = _flat_fixed_values(owner)
            values.update(
                {name: parameter_array[:, index] for name, index in free.items()}
            )
            derived = function(**values)
            first = min(free[name] for name in replaced)
            insert_at[first] = [
                (
                    name,
                    np.broadcast_to(
                        (
                            value.recentred(weights)
                            if isinstance(value, Periodic)
                            else np.asarray(value, dtype=float)
                        ),
                        (n_samples,),
                    ),
                )
                for name, value in derived.items()
            ]
            consumed.update(free[name] for name in replaced)

    out_names, out_superscripts, out_paths, out_columns, out_mask = [], [], [], [], []
    for i in range(len(names)):
        if i in insert_at:
            for name, column in insert_at[i]:
                out_names.append(name)
                out_superscripts.append(superscripts[i])
                out_paths.append(component_paths[i])
                out_columns.append(column)
                out_mask.append(True)
        if i in consumed:
            continue
        out_names.append(names[i])
        out_superscripts.append(superscripts[i])
        out_paths.append(component_paths[i])
        out_columns.append(parameter_array[:, i])
        out_mask.append(False)

    return DerivedParameters(
        names=out_names,
        superscripts=out_superscripts,
        component_paths=out_paths,
        parameter_array=(
            np.column_stack(out_columns) if out_columns else parameter_array
        ),
        derived_mask=out_mask,
    )
