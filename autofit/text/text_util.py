import logging

from autoconf import conf
from autofit.text import formatter as frm, samples_text


def results_to_file(samples, filename, during_analysis):
    """Output the full model.results file, which include the most-likely model, most-probable model at 1 and 3
    sigma confidence and information on the maximum log likelihood.
    Parameters
    ----------
    during_analysis : bool
        Whether the model.results are being written during the analysis or after the `NonLinearSearch` has finished.
    """

    results = []

    if hasattr(samples, "log_evidence"):
        if samples.log_evidence is not None:

            value = "{:.8f}".format(samples.log_evidence)
            results += [
                frm.add_whitespace(str0="Bayesian Evidence ", str1=value, whitespace=90)
            ]
            results += ["\n"]

    value = "{:.8f}".format(max(samples.log_likelihoods))
    results += [
        frm.add_whitespace(str0="Maximum Likelihood ", str1=value, whitespace=90)
    ]
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

            results += samples_text.summary(samples=samples, sigma=3.0, indent=4, line_length=90)
            results += ["\n"]
            results += samples_text.summary(samples=samples, sigma=1.0, indent=4, line_length=90)

        else:

            results += [
                "\n WARNING: The samples have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += samples_text.summary(samples=samples, sigma=1.0, indent=4, line_length=90)

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
    decimal_places = conf.instance["general"]["output"]["model_results_decimal_places"]
    return f"{{:.{decimal_places}f}}"