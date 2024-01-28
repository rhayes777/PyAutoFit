import logging

from autofit.text import formatter as frm

logger = logging.getLogger(__name__)


def values_from_samples(samples, median_pdf_model):
    if median_pdf_model:
        return samples.median_pdf(as_instance=False)
    return samples.max_log_likelihood(as_instance=False)


def summary(
    samples, sigma=3.0, median_pdf_model=True, indent=1, line_length=None
) -> str:
    """
    Create a string summarizing the results of the `NonLinearSearch` at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
    """

    values = values_from_samples(samples=samples, median_pdf_model=median_pdf_model)
    values_at_sigma = samples.values_at_sigma(sigma=sigma, as_instance=False)

    parameter_names = samples.model.parameter_names

    if line_length is None:
        line_length = len(max(parameter_names, key=len)) + 8

    sigma_formatter = frm.TextFormatter(indent=indent, line_length=line_length)

    for i, prior_path in enumerate(samples.model.unique_prior_paths):
        value_result = frm.value_result_string_from(
            parameter_name=parameter_names[i],
            value=values[i],
            values_at_sigma=values_at_sigma[i],
        )

        sigma_formatter.add(prior_path, value_result)

    return f"\n\nSummary ({sigma} sigma limits):\n\n{sigma_formatter.text}"


def derived_quantity_summary(
    samples,
    sigma=3.0,
    indent=1,
    median_pdf_model=True,
    line_length=None,
) -> str:
    """
    Create a string summarizing the results of the `NonLinearSearch` at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
    """

    instance = (
        samples.median_pdf() if median_pdf_model else samples.max_log_likelihood()
    )
    values = samples.derived_quantities_for_instance(instance)

    sigma_values = samples.values_at_sigma(sigma=sigma, as_instance=False)
    lower, upper = map(samples.model.instance_from_vector, zip(*sigma_values))
    values_at_sigma = list(
        zip(*map(samples.derived_quantities_for_instance, [lower, upper]))
    )

    parameter_names = list(zip(*samples.model.derived_quantities))[-1]

    if line_length is None:
        line_length = len(max(parameter_names, key=len)) + 8

    sigma_formatter = frm.TextFormatter(
        indent=indent,
        line_length=line_length,
    )

    for i, prior_path in enumerate(samples.model.derived_quantities):
        value_result = frm.value_result_string_from(
            parameter_name=parameter_names[i],
            value=values[i],
            values_at_sigma=values_at_sigma[i],
        )

        sigma_formatter.add(prior_path, value_result)

    return f"\n\nSummary ({sigma} sigma limits):\n\n{sigma_formatter.text}"


def latex(
    samples,
    median_pdf_model=True,
    sigma=3.0,
    name_to_label=True,
    include_name=True,
    include_quickmath=False,
    prefix="",
    suffix="",
) -> str:
    """
    Create a string summarizing the results of the `NonLinearSearch` at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
    """

    values = values_from_samples(samples=samples, median_pdf_model=median_pdf_model)
    errors_at_sigma = samples.errors_at_sigma(sigma=sigma, as_instance=False)

    table = []

    for i in range(samples.model.prior_count):
        label_value = frm.parameter_result_latex_from(
            parameter_name=samples.model.parameter_names[i],
            value=values[i],
            errors=errors_at_sigma[i],
            superscript=samples.model.superscripts[i],
            name_to_label=name_to_label,
            include_name=include_name,
            include_quickmath=include_quickmath,
        )

        table.append(f"{label_value}")

    table = "".join(table)[:-3]

    return f"{prefix}{table}{suffix}"
