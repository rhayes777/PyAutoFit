import configparser
import logging

from autofit import conf
from autofit.tools import text_util, text_formatter

logger = logging.getLogger(__name__)

def results_from_sigma(samples, sigma) -> str:
    """ Create a string summarizing the results of the non-linear search at an input sigma value.

    This function is used for creating the model.results files of a non-linear search.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
    try:
        lower_limits = samples.vector_at_lower_sigma(sigma=sigma)
        upper_limits = samples.vector_at_upper_sigma(sigma=sigma)
    except ValueError:
        return ""

    sigma_formatter = text_formatter.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):
        value = samples.format_str.format(samples.most_probable_vector[i])
        upper_limit = samples.format_str.format(upper_limits[i])
        lower_limit = samples.format_str.format(lower_limits[i])
        value = value + " (" + lower_limit + ", " + upper_limit + ")"
        sigma_formatter.add((prior_path, value))

    return "\n\nMost probable model ({} sigma limits):\n\n{}".format(
        sigma, sigma_formatter.text
    )


def latex_results_at_sigma(samples, sigma, format_str="{:.2f}") -> [str]:
    """Return the results of the non-linear search at an input sigma value as a string that is formated for simple
    copy and pasting in a LaTex document.

    Parameters
    ----------
    sigma : float
        The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
    format_str : str
        The formatting of the parameter string, e.g. how many decimal points to which the parameter is written.
    """

    labels = samples.param_labels
    most_probables = samples.most_probable_vector
    uppers = samples.vector_at_upper_sigma(sigma=sigma)
    lowers = samples.vector_at_lower_sigma(sigma=sigma)

    line = []

    for i in range(len(labels)):
        most_probable = format_str.format(most_probables[i])
        upper = format_str.format(uppers[i])
        lower = format_str.format(lowers[i])

        line += [
            labels[i]
            + " = "
            + most_probable
            + "^{+"
            + upper
            + "}_{-"
            + lower
            + "} & "
        ]

    return line


def format_str(samples) -> str:
    """The format string for the model.results file, describing to how many decimal points every parameter
    estimate is output in the model.results file.
    """
    decimal_places = conf.instance.general.get(
        "output", "model_results_decimal_places", int
    )
    return "{:." + str(decimal_places) + "f}"


def output_results(samples, during_analysis):
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
            results += text_util.label_and_value_string(
                label="Bayesian Evidence ",
                value=samples.log_evidence,
                whitespace=90,
                format_string="{:.8f}",
            )
            results += ["\n"]

    results += text_util.label_and_value_string(
        label="Maximum Likelihood ",
        value=samples.max_log_posterior,
        whitespace=90,
        format_string="{:.8f}",
    )
    results += ["\n\n"]

    results += ["Most Likely Model:\n\n"]
    most_likely = samples.max_log_likelihood_vector

    formatter = text_formatter.TextFormatter()

    for i, prior_path in enumerate(samples.model.unique_prior_paths):
        formatter.add((prior_path, samples.format_str.format(most_likely[i])))
    results += [formatter.text + "\n"]

    if samples.pdf_converged:

        results += samples.results_from_sigma(sigma=3.0)
        results += ["\n"]
        results += samples.results_from_sigma(sigma=1.0)

    else:

        results += [
            "\n WARNING: The chains have not converged enough to compute a PDF and model errors. \n "
            "The model below over estimates errors. \n\n"
        ]
        results += samples.results_from_sigma(sigma=1.0)

    results += ["\n\ninstances\n"]

    formatter = text_formatter.TextFormatter()

    for t in samples.model.path_float_tuples:
        formatter.add(t)

    results += ["\n" + formatter.text]

    text_util.output_list_of_strings_to_file(
        file=samples.paths.file_results, list_of_strings=results
    )