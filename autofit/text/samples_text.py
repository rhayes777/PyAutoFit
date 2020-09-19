import logging

from autofit.text import formatter as frm

logger = logging.getLogger(__name__)


def values_from_samples(samples, median_pdf_model):

    if median_pdf_model:
        return samples.median_pdf_vector
    return samples.max_log_likelihood_vector

def summary(samples, sigma=3.0, median_pdf_model=True, name_to_label=True) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

    values = values_from_samples(samples=samples, median_pdf_model=median_pdf_model)
    values_at_sigma = samples.vector_at_sigma(sigma=sigma)

    sigma_formatter = frm.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):

        value = frm.parameter_result_string_from(
            parameter_name=samples.model.parameter_names[i],
            value=values[i],
            values_at_sigma=values_at_sigma[i],
            name_to_label=name_to_label,
        )

        sigma_formatter.add((prior_path, value))

    return "\n\nMedian PDF model Summary ({} sigma limits):\n\n{}".format(
        sigma, sigma_formatter.text
    )


def table(samples, median_pdf_model=True, sigma=3.0, name_to_label=True) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

    values = values_from_samples(samples=samples, median_pdf_model=median_pdf_model)
    values_at_sigma = samples.vector_at_sigma(sigma=sigma)

    table = []

    for i in range(samples.model.prior_count):

        label_value = frm.parameter_result_string_from(
            parameter_name=samples.model.parameter_names[i],
            value=values[i],
            values_at_sigma=values_at_sigma[i],
            subscript=samples.model.subscripts[i],
            name_to_label=name_to_label,
        )
        table.append(f"{label_value} & ")

    table = "".join(table)[:-3]

    return "\n\nMedian PDF model Table ({} sigma limits):\n\n{}".format(sigma, table)


def latex(samples, median_pdf_model=True, sigma=3.0, name_to_label=True) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

    values = values_from_samples(samples=samples, median_pdf_model=median_pdf_model)
    errors_at_sigma = samples.error_vector_at_sigma(sigma=sigma)

    table = []

    for i in range(samples.model.prior_count):

        label_value = frm.parameter_result_latex_from(
            parameter_name=samples.model.parameter_names[i],
            value=values[i],
            errors=errors_at_sigma[i],
            subscript=samples.model.subscripts[i],
            name_to_label=name_to_label,
        )

        table.append(f"{label_value}")


    table = "".join(table)[:-3]

    return "\n\nMedian PDF model Table ({} sigma limits):\n\n{}".format(sigma, table)



