"""
Latent parameters: a model component's own re-parameterisation of some of its
free parameters, for plotting and reporting.

Many components are *sampled* in one parameterisation and *read* in another: a
shear is fit as ``(gamma_1, gamma_2)`` but has a magnitude and an angle, an
ellipse as two elliptical components but has an axis ratio and a position
angle, a multipole as two components but has a strength and a phase. A component
class declares the conversion as a class attribute::

    class ExternalShear:
        latent_parameters = {
            ("gamma_1", "gamma_2"): shear_magnitude_and_angle_dict_from,
        }

keyed by the names of the free parameters it replaces and valued by a function
that takes every parameter of the component by name (the free ones as arrays
over the samples, the fixed ones as floats) and returns a ``dict`` of latent
name to array. The declarations of a class's bases are inherited, so a subclass
adds to them rather than restating them.

``LatentParameters`` applies those declarations to an array of samples: the
replaced parameters go, the latent ones take their place (at the position of
the first replaced one, in model order), everything else passes through
unchanged, and each latent parameter carries the superscript of the component
it came from. A declaration whose replaced parameters are not all free in this
model is skipped, since there is nothing to convert.

A prior shared between components (a power law's ellipticity linked into its
multipoles) is seen by the declarations of every component holding it, so each
can convert it; the same conversion of the same priors is made once, under the
component the priors' samples column is named after.

An angle is periodic, and a conversion fixes where its branch cut falls: a
position angle in ``[-45, 135)`` whose posterior peaks near 135 is split in two,
and its corner plot shows a double peak that is not there. A conversion returns
such a value as ``Periodic(angle, period)``. For a plot (``recentre=True``) the
samples are recentred on their (weighted) circular mean, so the cut sits
opposite the peak. The latent output (``latent.csv``) keeps the conversion's own
range, so the same parameter has the same range in every fit and fits can be
compared; a posterior straddling the cut is then split there, and its median and
errors in ``latent.results`` with it.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from autofit.text import formatter as frm

LatentFunction = Callable[..., Dict[str, np.ndarray]]


class Periodic:
    def __init__(self, values, period: float):
        """
        A latent parameter that is periodic (an angle), returned by a conversion so
        the samples are recentred on their circular mean; see the module docstring.

        Parameters
        ----------
        values
            The latent values over the samples, in whatever range the conversion gives.
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


def latent_parameters_for_class(cls) -> Dict[Tuple[str, ...], LatentFunction]:
    """
    The latent-parameter declarations of a class and all of its bases, the most
    latent class last so it can override an entry of a base.
    """
    declarations = {}
    for base in reversed(getattr(cls, "__mro__", (cls,))):
        declarations.update(base.__dict__.get("latent_parameters", {}) or {})
    return declarations


class LatentParameters:
    def __init__(
        self,
        names: List[str],
        superscripts: List[Optional[str]],
        component_paths: List[Tuple[str, ...]],
        parameter_array: np.ndarray,
        latent_mask: List[bool],
        paths: Optional[List[Tuple[str, ...]]] = None,
    ):
        """
        Parameters of a model after each component's declared conversions are
        applied to an array of samples; see the module docstring.

        Parameters
        ----------
        names
            The name of every parameter of the latent set, in order.
        superscripts
            The superscript of each, that of the component it belongs to.
        component_paths
            The named-component path of each (see ``AbstractPriorModel.component_paths``).
        parameter_array
            The samples of the latent set, shape ``(n_samples, len(names))``.
        latent_mask
            Whether each parameter was latent (``True``) or passed through.
        paths
            The full model path of each (``galaxies.lens.shear.magnitude``), a
            latent one under the component that declared it; the key it has in
            the latent output (``latent.csv``).
        """
        self.names = names
        self.superscripts = superscripts
        self.component_paths = component_paths
        self.parameter_array = parameter_array
        self.latent_mask = latent_mask
        self.paths = paths

    @property
    def has_latent(self) -> bool:
        """Whether any declaration was applied, so the set differs from the model's own."""
        return any(self.latent_mask)

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
        """As ``AbstractPriorModel.parameter_indices_by_component``, over the latent set."""
        from autofit.mapper.prior_model.abstract import indices_by_top_level_component

        return indices_by_top_level_component(self.component_paths)

    @property
    def parameter_indices_by_latent_component(self) -> Dict[str, List[int]]:
        """
        ``parameter_indices_by_component`` restricted to the components with at
        least one latent parameter: the plot of a component without one would
        repeat its plain corner plot.
        """
        return {
            name: indices
            for name, indices in self.parameter_indices_by_component.items()
            if any(self.latent_mask[i] for i in indices)
        }

    @property
    def latent_items(self) -> List[Tuple[Tuple[str, ...], np.ndarray]]:
        """
        The latent parameters alone, as ``(path, samples)`` pairs, which is what
        the latent output (``latent.csv``) takes from the model.
        """
        if self.paths is None:
            raise ValueError("This LatentParameters was built without paths.")
        return [
            (path, self.parameter_array[:, i])
            for i, (path, is_latent) in enumerate(zip(self.paths, self.latent_mask))
            if is_latent
        ]


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


def _owner_path(prior_path, leaf: str) -> Tuple[str, ...]:
    """
    The model path of the component owning a prior, from the prior's full path:
    the leaf is stripped, and so is the tuple it sits in (``centre.centre_0``).
    """
    prior_path = tuple(prior_path)
    owner = prior_path[:-1]
    if owner and leaf.startswith(f"{owner[-1]}_"):
        owner = owner[:-1]
    return owner


def latent_parameters_from(
    model, parameter_array: np.ndarray, weights=None, recentre: bool = False
) -> LatentParameters:
    """
    Apply every component's declared conversions to an array of samples of the
    model; see the module docstring. Called through
    ``AbstractPriorModel.latent_parameters_from``.

    Parameters
    ----------
    weights
        The weight of each sample, used with ``recentre`` to centre a
        ``Periodic`` latent parameter on its posterior rather than on the prior
        volume; equal by default.
    recentre
        Whether to recentre each ``Periodic`` latent parameter on its circular
        mean (for a plot), rather than keep the range the conversion gives it.
    """
    from autofit.mapper.prior_model.abstract import superscripts_from_component_paths

    parameter_array = np.asarray(parameter_array)
    n_samples = parameter_array.shape[0]

    holders = model._prior_holders
    names = model.parameter_names
    superscripts = model.superscripts
    component_paths = model.component_paths
    paths = model.unique_prior_paths

    # Every component holding a free parameter, with what it holds by leaf name:
    # the model index and the holder (prior path, component path) there. A
    # shared prior is held by each component it sits in.
    indices_by_owner = {}
    for i, prior_holders in enumerate(holders):
        for holder in prior_holders:
            prior_path, component_path, owner, leaf = holder
            free = indices_by_owner.setdefault(id(owner), (owner, {}))[1]
            free.setdefault(leaf, (i, prior_path, component_path))

    # A component converts its own priors before one that only shares them, so a
    # conversion made by both is filed under the one the samples are named after.
    def borrowed(entry):
        owner, free = entry
        return sum(holders[i][0][2] is not owner for i, _, _ in free.values())

    # Resolve every applicable declaration first: which indices it consumes and
    # what it yields, keyed by the index it is inserted at.
    insert_at = {}
    consumed = set()
    made = set()
    for owner, free in sorted(indices_by_owner.values(), key=borrowed):
        cls = getattr(owner, "cls", None)
        if cls is None:
            continue
        for replaced, function in latent_parameters_for_class(cls).items():
            if not all(name in free for name in replaced):
                continue
            replaced_indices = tuple(free[name][0] for name in replaced)
            if (function, replaced_indices) in made:
                continue
            made.add((function, replaced_indices))
            values = _flat_fixed_values(owner)
            values.update(
                {
                    name: parameter_array[:, index]
                    for name, (index, _, _) in free.items()
                }
            )
            latent = function(**values)
            first_name = min(replaced, key=lambda name: free[name][0])
            first, prior_path, component_path = free[first_name]
            insert_at.setdefault(first, []).extend(
                (
                    _owner_path(prior_path, first_name) + (name,),
                    component_path,
                    name,
                    np.broadcast_to(
                        (
                            (value.recentred(weights) if recentre else value.values)
                            if isinstance(value, Periodic)
                            else np.asarray(value, dtype=float)
                        ),
                        (n_samples,),
                    ),
                )
                for name, value in latent.items()
            )
            consumed.update(replaced_indices)

    out_names, out_superscripts, out_paths, out_columns, out_mask = [], [], [], [], []
    out_full_paths = []
    for i in range(len(names)):
        for full_path, component_path, name, column in insert_at.get(i, []):
            out_names.append(name)
            out_superscripts.append(superscripts[i])
            out_paths.append(component_path)
            out_full_paths.append(full_path)
            out_columns.append(column)
            out_mask.append(True)
        if i in consumed:
            continue
        out_names.append(names[i])
        out_superscripts.append(superscripts[i])
        out_paths.append(component_paths[i])
        out_full_paths.append(tuple(paths[i]))
        out_columns.append(parameter_array[:, i])
        out_mask.append(False)

    # A latent parameter of a component that only shares the prior is labelled
    # by that component, not by the one the prior's column is named after.
    labels = superscripts_from_component_paths(component_paths + out_paths)
    out_superscripts = [
        labels[path] if path else superscript
        for path, superscript in zip(out_paths, out_superscripts)
    ]

    return LatentParameters(
        names=out_names,
        superscripts=out_superscripts,
        component_paths=out_paths,
        parameter_array=(
            np.column_stack(out_columns) if out_columns else parameter_array
        ),
        latent_mask=out_mask,
        paths=out_full_paths,
    )
