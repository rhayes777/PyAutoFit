import logging

from autoconf import conf
from autofit.text import formatter as frm

logger = logging.getLogger(__name__)


def parameter_results_at_sigma_summary(samples, sigma) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
    values_at_sigma = samples.vector_at_sigma(sigma=sigma)

    sigma_formatter = frm.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):

        value = frm.parameter_result_string_from(
            parameter_name=samples.model.parameter_names[i],
            value=samples.median_pdf_vector[i],
            values_at_sigma=values_at_sigma[i],
        )

        sigma_formatter.add((prior_path, value))

    return "\n\nMedian PDF model Summary ({} sigma limits):\n\n{}".format(
        sigma, sigma_formatter.text
    )


def parameter_results_at_sigma_table(samples, sigma, name_to_label=True) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

    values_at_sigma = samples.vector_at_sigma(sigma=sigma)

    table = []

    for i in range(samples.model.prior_count):

        label_value = frm.parameter_result_string_from(
            parameter_name=samples.model.parameter_names[i],
            value=samples.median_pdf_vector[i],
            values_at_sigma=values_at_sigma[i],
            subscript=samples.model.subscripts[i],
            name_to_label=name_to_label,
        )
        table.append(f"{label_value} & ")

    table = "".join(table)[:-3]

    return "\n\nMedian PDF model Table ({} sigma limits):\n\n{}".format(sigma, table)


def parameter_results_at_sigma_latex(samples, sigma, name_to_label=True) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

    errors_at_sigma = samples.error_vector_at_sigma(sigma=sigma)

    table = []

    for i in range(samples.model.prior_count):

        label_value = frm.parameter_result_latex_from(
            parameter_name=samples.model.parameter_names[i],
            value=samples.median_pdf_vector[i],
            errors=errors_at_sigma[i],
            subscript=samples.model.subscripts[i],
            name_to_label=name_to_label,
        )

        table.append(f"{label_value}")


    table = "".join(table)[:-3]

    return "\n\nMedian PDF model Table ({} sigma limits):\n\n{}".format(sigma, table)


def results_to_file(samples, filename, during_analysis):
    """Output the full model.results file, which include the most-likely model, most-probable model at 1 and 3
    sigma confidence and information on the maximum log likelihood.

    Parameters
    ----------
    during_analysis : bool
        Whether the model.results are being written during the analysis or after the non-linear search has finished.
    """

    results = []

    if hasattr(samples, "log_evidence"):
        if samples.log_evidence is not None:

            value = "{:.8f}".format(samples.log_evidence)
            results += [frm.add_whitespace(str0="Bayesian Evidence ", str1=value, whitespace=90)]
            results += ["\n"]

    value = "{:.8f}".format(max(samples.log_likelihoods))
    results += [frm.add_whitespace(str0="Maximum Likelihood ", str1=value, whitespace=90)]
    results += ["\n\n"]

    results += ["Maximum Log Likelihood Model:\n\n"]

    formatter = frm.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):
        formatter.add(
            (prior_path, format_str().format(samples.max_log_likelihood_vector[i]))
        )
    results += [formatter.text + "\n"]

    if hasattr(samples, "pdf_converged"):

        if samples.pdf_converged:

            results += parameter_results_at_sigma_summary(
                samples=samples, sigma=3.0
            )
            results += ["\n"]
            results += parameter_results_at_sigma_summary(
                samples=samples, sigma=1.0
            )

        else:

            results += [
                "\n WARNING: The samples have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += parameter_results_at_sigma_summary(
                samples=samples, sigma=1.0
            )

        results += ["\n\ninstances\n"]

    formatter = frm.TextFormatter()

    for t in samples.model.path_float_tuples:
        formatter.add(t)

    results += ["\n" + formatter.text]

    frm.output_list_of_strings_to_file(file=filename, list_of_strings=results)


def search_summary_from_samples(samples) -> [str]:

    line = [f"Total Samples = {samples.total_samples}\n"]
    if hasattr(samples, "total_accepted_samples"):
        line.append(f"Total Accepted Samples = {samples.total_accepted_samples}\n")
        line.append(f"Acceptance Ratio = {samples.acceptance_ratio}\n")
    if samples.time is not None:
        line.append(f"Time To Run = {samples.time}\n")
    return line


def search_summary_to_file(samples, filename):

    summary = search_summary_from_samples(samples=samples)

    frm.output_list_of_strings_to_file(file=filename, list_of_strings=summary)


def format_str() -> str:
    """The format string for the model.results file, describing to how many decimal points every parameter
    estimate is output in the model.results file.
    """
    decimal_places = conf.instance.general.get(
        "output", "model_results_decimal_places", int
    )
    return f"{{:.{decimal_places}f}}"
