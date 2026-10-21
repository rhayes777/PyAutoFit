import numpy as np
import warnings

from autonerves import conf

from autofit.non_linear.plot.plot_util import (
    skip_in_test_mode,
    log_plot_exception,
    output_figure,
    accepted_kwarg_names,
    checked_kwargs,
)


@skip_in_test_mode
@log_plot_exception
def corner_anesthetic(
    samples,
    path=None,
    filename="corner_anesthetic",
    format="show",
    parameter_indices=None,
    latent=False,
    **kwargs,
):
    """
    Corner plot of the posterior via anesthetic, over every parameter of the model
    or, with ``parameter_indices``, over a subset of them (the parameters of one
    model component, say) so that a many-parameter fit stays readable.

    Parameters
    ----------
    samples
        The samples of the non-linear search, whose ``model`` labels the axes.
    parameter_indices
        The indices (in model order, or in the latent set's order when
        ``latent``) of the parameters to plot; all by default.
    latent
        Plot the model's latent parameters instead of its free ones: each
        component's declared conversion is applied to the samples (a shear's
        components become its magnitude and angle, say) and parameters without
        one pass through; see ``AbstractPriorModel.latent_parameters_from``.
    """
    config_dict = conf.instance["visualize"]["plots_settings"]["corner_anesthetic"]

    from anesthetic.samples import NestedSamples
    from anesthetic import make_2d_axes
    import matplotlib.pylab as pylab

    params = {"font.size": int(config_dict["fontsize"])}
    pylab.rcParams.update(params)

    model = samples.model

    parameter_array = np.asarray(samples.parameter_lists)

    if latent:
        latent_parameters = model.latent_parameters_from(
            parameter_array, weights=samples.weight_list, recentre=True
        )
        labels = latent_parameters.parameter_labels_with_superscripts_latex
        parameter_array = latent_parameters.parameter_array
    else:
        labels = model.parameter_labels_with_superscripts_latex

    if parameter_indices is not None:
        parameter_indices = list(parameter_indices)
        labels = [labels[i] for i in parameter_indices]
        parameter_array = parameter_array[:, parameter_indices]

    figsize = (
        len(labels) * config_dict["figsize_per_parammeter"],
        len(labels) * config_dict["figsize_per_parammeter"],
    )

    # anesthetic keys its columns by name, but LaTeX labels repeat whenever a model
    # has several components of one type -- every Gaussian of a multi-Gaussian basis
    # is labelled $\epsilon_{1}^{g}$. A repeated key makes plot_2d raise KeyError,
    # which @log_plot_exception reports as an unconverged posterior, so the plot
    # silently never appears. Key the columns by position, which is always unique,
    # and pass the LaTeX labels for display only.
    columns = [f"p{i}" for i in range(len(labels))]

    nested_samples = NestedSamples(
        parameter_array,
        weights=samples.weight_list,
        columns=columns,
        labels=labels,
    )

    try:
        from pandas.errors import SettingWithCopyWarning
    except ImportError:  # pandas >= 2.2 removed SettingWithCopyWarning
        SettingWithCopyWarning = None

    if SettingWithCopyWarning is not None:
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    # ``kwargs`` splits by destination: what shapes the figure goes to
    # ``make_2d_axes``, everything else styles the plot via ``plot_2d``, which
    # forwards what it does not name straight to matplotlib — matplotlib raises
    # on an unknown property, so unlike ``corner`` there is no silent sink here
    # to guard against and the only check needed is that the caller is not
    # overriding the sample array.
    #
    # ``figsize`` / ``facecolor`` / ``dpi`` are named explicitly because
    # ``make_2d_axes`` takes them through its own ``**fig_kw`` rather than
    # declaring them — including the two this function computes below, which a
    # caller must be able to override.
    axes_names = (accepted_kwarg_names(make_2d_axes) - {"params"}) | {
        "figsize",
        "facecolor",
        "dpi",
    }
    kwargs = checked_kwargs(
        kwargs,
        reserved=("data", "weights", "columns", "labels"),
        target="anesthetic",
    )

    axes_settings = dict(figsize=figsize, facecolor=config_dict["facecolor"])
    axes_settings.update(
        {key: value for key, value in kwargs.items() if key in axes_names}
    )

    plot_settings = dict(alpha=config_dict["alpha"], label="posterior")
    plot_settings.update(
        {key: value for key, value in kwargs.items() if key not in axes_names}
    )

    fig, axes = make_2d_axes(
        columns,
        labels=nested_samples.get_labels_map(),
        **axes_settings,
    )

    if SettingWithCopyWarning is not None:
        warnings.filterwarnings("default", category=SettingWithCopyWarning)

    nested_samples.plot_2d(
        axes,
        **plot_settings,
    )
    axes.iloc[-1, 0].legend(
        bbox_to_anchor=(len(axes) / 2, len(axes)),
        loc="lower center",
        ncols=2,
    )

    output_figure(path=path, filename=filename, format=format)
