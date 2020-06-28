import logging

from autoconf import conf
from autofit.text import formatter as frm, model_text

logger = logging.getLogger(__name__)


def results_at_sigma_from_samples(samples, sigma) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
    lower_limits = samples.vector_at_lower_sigma(sigma=sigma)
    upper_limits = samples.vector_at_upper_sigma(sigma=sigma)

    sigma_formatter = frm.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):
        value = format_str().format(samples.median_pdf_vector[i])
        upper_limit = format_str().format(upper_limits[i])
        lower_limit = format_str().format(lower_limits[i])
        value = f"{value} ({lower_limit}, {upper_limit})"
        sigma_formatter.add((prior_path, value))

    return "\n\nMedian PDF model ({} sigma limits):\n\n{}".format(
        sigma, sigma_formatter.text
    )

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
            results += frm.label_and_value_string(
                label="Bayesian Evidence ",
                value=samples.log_evidence,
                whitespace=90,
                format_string="{:.8f}",
            )
            results += ["\n"]

    results += frm.label_and_value_string(
        label="Maximum Likelihood ",
        value=max(samples.log_likelihoods),
        whitespace=90,
        format_string="{:.8f}",
    )
    results += ["\n\n"]

    results += ["Maximum Log Likelihood Model:\n\n"]

    formatter = frm.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):
        formatter.add((prior_path, format_str().format(samples.max_log_likelihood_vector[i])))
    results += [formatter.text + "\n"]

    if hasattr(samples, "pdf_converged"):

        if samples.pdf_converged:

            results += results_at_sigma_from_samples(samples=samples, sigma=3.0)
            results += ["\n"]
            results += results_at_sigma_from_samples(samples=samples, sigma=1.0)

        else:

            results += [
                "\n WARNING: The samples have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += results_at_sigma_from_samples(samples=samples, sigma=1.0)

        results += ["\n\ninstances\n"]

    formatter = frm.TextFormatter()

    for t in samples.model.path_float_tuples:
        formatter.add(t)

    results += ["\n" + formatter.text]

    frm.output_list_of_strings_to_file(
        file=filename, list_of_strings=results
    )


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

    labels = model_text.parameter_labels_from_model(model=samples.model)
    median_pdfs = samples.median_pdf_vector
    uppers = samples.vector_at_upper_sigma(sigma=sigma)
    lowers = samples.vector_at_lower_sigma(sigma=sigma)

    line = []

    for i in range(len(labels)):
        median_pdf = format_str.format(median_pdfs[i])
        upper = format_str.format(uppers[i])
        lower = format_str.format(lowers[i])

        line += [
            labels[i]
            + " = "
            + median_pdf
            + "^{+"
            + upper
            + "}_{-"
            + lower
            + "} & "
        ]

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

    frm.output_list_of_strings_to_file(
        file=filename, list_of_strings=summary
    )

def format_str() -> str:
    """The format string for the model.results file, describing to how many decimal points every parameter
    estimate is output in the model.results file.
    """
    decimal_places = conf.instance.general.get(
        "output", "model_results_decimal_places", int
    )
    return f"{{:.{decimal_places}f}}"


