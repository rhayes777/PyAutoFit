import logging

from autoconf import conf
from autofit.text import formatter as frm

logger = logging.getLogger(__name__)


def median_pdf_with_errors_at_sigma_summary(samples, sigma) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
    values_at_sigma = samples.vector_at_sigma(sigma=sigma)

    sigma_formatter = frm.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):

        value = frm.value_with_limits_string(
            parameter_name=samples.model.parameter_names[i],
            value=samples.median_pdf_vector[i],
            values_at_sigma=values_at_sigma[i],
        )

        sigma_formatter.add((prior_path, value))

    return "\n\nMedian PDF model Summary ({} sigma limits):\n\n{}".format(
        sigma, sigma_formatter.text
    )


def median_pdf_with_errors_at_sigma_table(samples, sigma, name_to_label=True) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""

    values_at_sigma = samples.vector_at_sigma(sigma=sigma)

    table = []

    for i, prior_path in enumerate(samples.model.unique_prior_paths):

        label_value = frm.parameter_result_string_from(
            parameter_name=samples.model.parameter_names[i],
            value=samples.median_pdf_vector[i],
            values_at_sigma=values_at_sigma[i],
            subscript=samples.model.subscripts[i],
            whitespace=12,
            name_to_label=name_to_label,
        )
        table.append(f"{label_value} | ")

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
            results += frm.parameter_result_string_from(
                parameter_name="Bayesian Evidence ",
                value=samples.log_evidence,
                whitespace=90,
                format_string="{:.8f}",
            )
            results += ["\n"]

    results += frm.parameter_result_string_from(
        parameter_name="Maximum Likelihood ",
        value=max(samples.log_likelihoods),
        whitespace=90,
        format_string="{:.8f}",
    )
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

            results += median_pdf_with_errors_at_sigma_summary(
                samples=samples, sigma=3.0
            )
            results += ["\n"]
            results += median_pdf_with_errors_at_sigma_summary(
                samples=samples, sigma=1.0
            )

        else:

            results += [
                "\n WARNING: The samples have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += median_pdf_with_errors_at_sigma_summary(
                samples=samples, sigma=1.0
            )

        results += ["\n\ninstances\n"]

    formatter = frm.TextFormatter()

    for t in samples.model.path_float_tuples:
        formatter.add(t)

    results += ["\n" + formatter.text]

    frm.output_list_of_strings_to_file(file=filename, list_of_strings=results)


def latex_results_at_sigma_from_samples(samples, sigma, format_str="{:.2f}") -> [str]:
    """Return the results of the non-linear search at an input sigma value as a string that is formated for simple
    copy and pasting in a LaTex document.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
    format_str : str
        The formatting of the parameter string, e.g. how many decimal points to which the parameter is written.
    """

    labels = samples.model.parameter_labels
    subscripts = samples.model.subscripts
    labels = [
        f"{label}_{{\\mathrm{{{subscript}}}}}"
        for label, subscript in zip(labels, subscripts)
    ]
    median_pdfs = samples.median_pdf_vector
    uppers = samples.vector_at_upper_sigma(sigma=sigma)
    lowers = samples.vector_at_lower_sigma(sigma=sigma)

    line = []

    for i in range(len(labels)):
        median_pdf = format_str.format(median_pdfs[i])
        upper = format_str.format(uppers[i])
        lower = format_str.format(lowers[i])

        line += [f"{labels[i]} = {median_pdf}^{{+{upper}}}_{{-{lower}}} & "]

    return line


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
    decimal_places = conf.instance["general"]["output"]["model_results_decimal_places"]
    return f"{{:.{decimal_places}f}}"
