import logging
import os
from configparser import NoOptionError

from autofit import conf
from autofit.mapper import model
from autofit.tools import text_formatter, text_util

logger = logging.getLogger(__name__)


class AbstractOutput:
    def __init__(self, model, paths):
        """The *Output* classes in **PyAutoFit** provide an interface between the results of a non-linear search (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the non-linear search and return information on the likelihoods, errors, etc.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc."""
        self.model = model
        self.paths = paths

    @property
    def total_samples(self) -> int:
        """The total number of samples performed by the non-linear search."""
        raise NotImplementedError()

    @property
    def maximum_log_likelihood(self) -> float:
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model."""
        raise NotImplementedError()

    @property
    def most_likely_vector(self) -> [float]:
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log-likelihood), returned
        as a list of values."""
        raise NotImplementedError()

    @property
    def most_likely_instance(self) -> model.ModelInstance:
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log-likelihood), returned
        as a model instance."""
        return self.model.instance_from_vector(vector=self.most_likely_vector)

    @property
    def most_probable_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values."""
        raise NotImplementedError()

    @property
    def most_probable_instance(self) -> model.ModelInstance:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a model instance."""
        return self.model.instance_from_vector(vector=self.most_probable_vector)

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        raise NotImplementedError()

    def vector_at_upper_sigma(self, sigma) -> [float]:
        """The upper value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[1], self.vector_at_sigma(sigma)))

    def vector_at_lower_sigma(self, sigma) -> [float]:
        """The lower value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[0], self.vector_at_sigma(sigma)))

    def instance_at_sigma(self, sigma) -> model.ModelInstance:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as a list of model instances corresponding to the lower and upper values estimated from the PDF.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        return self.model.instance_from_vector(
            vector=self.vector_at_sigma(sigma=sigma), assert_priors_in_limits=False
        )

    def instance_at_upper_sigma(self, sigma) -> model.ModelInstance:
        """The upper value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.vector_at_upper_sigma(sigma=sigma),
            assert_priors_in_limits=False,
        )

    def instance_at_lower_sigma(self, sigma) -> model.ModelInstance:
        """The lower value of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.vector_at_lower_sigma(sigma=sigma),
            assert_priors_in_limits=False,
        )

    def error_vector_at_sigma(self, sigma) -> [float]:
        """ The value of every error after marginalization in 1D at an input sigma value of the probability density
        function (PDF), returned as two lists of values corresponding to the lower and upper errors.

        For example, if sigma is 1.0, the errors marginalized at 31.7% and 68.2% percentiles of each PDF is returned.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        lowers = self.vector_at_lower_sigma(sigma=sigma)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def error_vector_at_upper_sigma(self, sigma) -> [float]:
        """The upper error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        return list(
            map(
                lambda upper, most_probable: upper - most_probable,
                uppers,
                self.most_probable_vector,
            )
        )

    def error_vector_at_lower_sigma(self, sigma) -> [float]:
        """The lower error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a list.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        lowers = self.vector_at_lower_sigma(sigma=sigma)
        return list(
            map(
                lambda lower, most_probable: most_probable - lower,
                lowers,
                self.most_probable_vector,
            )
        )

    def error_instance_at_sigma(self, sigma) -> model.ModelInstance:
        """ The error of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as a list of model instances corresponding to the lower and upper errors.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        return self.model.instance_from_vector(
            vector=self.error_vector_at_sigma(sigma=sigma)
        )

    def error_instance_at_upper_sigma(self, sigma) -> model.ModelInstance:
        """The upper error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.error_vector_at_upper_sigma(sigma=sigma)
        )

    def error_instance_at_lower_sigma(self, sigma) -> model.ModelInstance:
        """The lower error of every parameter marginalized in 1D at an input sigma value of its probability density
        function (PDF), returned as a model instance.

        See *vector_at_sigma* for a full description of how the parameters at sigma are computed.

        Parameters
        -----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """
        return self.model.instance_from_vector(
            vector=self.error_vector_at_lower_sigma(sigma=sigma)
        )

    def gaussian_priors_at_sigma(self, sigma) -> [list]:
        """*GaussianPrior*s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent phase, where:

         - The mean is given by their most-probable values (using *most_probable_vector*).
         - Their errors are computed at an input sigma value (using *errors_at_sigma*).

        Parameters
        -----------
        sigma : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.most_probable_vector
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        lowers = self.vector_at_lower_sigma(sigma=sigma)

        # noinspection PyArgumentList
        sigmas = list(
            map(
                lambda mean, upper, lower: max([upper - mean, mean - lower]),
                means,
                uppers,
                lowers,
            )
        )

        return list(map(lambda mean, sigma: (mean, sigma), means, sigmas))

    def likelihood_from_sample_index(self, sample_index) -> float:
        """The likelihood of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        raise NotImplementedError()

    def vector_from_sample_index(self, sample_index) -> [float]:
        """The parameters of an individual sample of the non-linear search, returned as a 1D list.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        raise NotImplementedError()

    def instance_from_sample_index(self, sample_index) -> model.ModelInstance:
        """The parameters of an individual saple of the non-linear search, returned as a model instance.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        model_parameters = self.vector_from_sample_index(sample_index=sample_index)

        return self.model.instance_from_vector(vector=model_parameters)

    def offset_vector_from_input_vector(self, input_vector) -> [float]:
        """ The values of an input_vector offset by the *most_probable_vector* (the PDF medians).

        If the 'true' values of a model are known and input as the *input_vector*, this function returns the results
        of the non-linear search as values offset from the 'true' model. For example, a value 0.0 means the non-linear
        search estimated the model parameter value correctly.

        Parameters
        ----------

        input_vector : list
            A 1D list of values the most-probable model is offset by. This list must have the same dimensions as free
            parameters in the model-fit.
        """
        return list(
            map(
                lambda input, most_probable: most_probable - input,
                input_vector,
                self.most_probable_vector,
            )
        )

    def results_from_sigma(self, sigma) -> str:
        """ Create a string summarizing the results of the non-linear search at an input sigma value.

        This function is used for creating the model.results files of a non-linear search.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        try:
            lower_limits = self.vector_at_lower_sigma(sigma=sigma)
            upper_limits = self.vector_at_upper_sigma(sigma=sigma)
        except ValueError:
            return ""

        sigma_formatter = text_formatter.TextFormatter()

        for i, prior_path in enumerate(self.model.unique_prior_paths):
            value = self.format_str.format(self.most_probable_vector[i])
            upper_limit = self.format_str.format(upper_limits[i])
            lower_limit = self.format_str.format(lower_limits[i])
            value = value + " (" + lower_limit + ", " + upper_limit + ")"
            sigma_formatter.add((prior_path, value))

        return "\n\nMost probable model ({} sigma limits):\n\n{}".format(
            sigma, sigma_formatter.text
        )

    @property
    def param_labels(self) -> [str]:
        """A list of every parameter's label, used by *GetDist* for model estimation and visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""

        paramnames_labels = []
        prior_class_dict = self.model.prior_class_dict
        prior_prior_model_dict = self.model.prior_prior_model_dict

        for prior_name, prior in self.model.prior_tuples_ordered_by_id:
            try:
                param_string = conf.instance.label.label(prior_name)
            except NoOptionError:
                logger.warning(
                    f"No label provided for {prior_name}. Using prior name instead."
                )
                param_string = prior_name
            prior_model = prior_prior_model_dict[prior]
            cls = prior_class_dict[prior]
            cls_string = "{}{}".format(
                conf.instance.label.subscript(cls), prior_model.component_number + 1
            )
            param_label = "{}_{{\\mathrm{{{}}}}}".format(param_string, cls_string)
            paramnames_labels.append(param_label)

        return paramnames_labels

    def create_paramnames_file(self):
        """Create the param_names file listing every parameter's label and Latex tag, which is used for *GetDist*
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""
        paramnames_names = self.model.param_names
        paramnames_labels = self.param_labels

        paramnames = []

        for i in range(self.model.prior_count):
            line = text_util.label_and_label_string(
                label0=paramnames_names[i], label1=paramnames_labels[i], whitespace=70
            )
            paramnames += [line + "\n"]

        text_util.output_list_of_strings_to_file(
            file=self.paths.file_param_names, list_of_strings=paramnames
        )

    def save_model_info(self):
        """Save the model.info file, which summarizes every parameter and prior."""

        try:
            os.makedirs(self.paths.backup_path)
        except FileExistsError:
            pass

        self.create_paramnames_file()

        with open(self.paths.file_model_info, "w+") as f:
            f.write(self.model.info)

    def latex_results_at_sigma(self, sigma, format_str="{:.2f}") -> [str]:
        """Return the results of the non-linear search at an input sigma value as a string that is formated for simple
        copy and pasting in a LaTex document.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        format_str : str
            The formatting of the parameter string, e.g. how many decimal points to which the parameter is written.
        """

        labels = self.param_labels
        most_probables = self.most_probable_vector
        uppers = self.vector_at_upper_sigma(sigma=sigma)
        lowers = self.vector_at_lower_sigma(sigma=sigma)

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

    @property
    def format_str(self) -> str:
        """The format string for the model.results file, describing to how many decimal points every parameter
        estimate is output in the model.results file.
        """
        decimal_places = conf.instance.general.get(
            "output", "model_results_decimal_places", int
        )
        return "{:." + str(decimal_places) + "f}"

    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/
        """
        raise NotImplementedError()

    @property
    def pdf_converged(self):
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use."""
        raise NotImplementedError()

    def output_results(self, during_analysis):
        """Output the full model.results file, which include the most-likely model, most-probable model at 1 and 3
        sigma confidence and information on the maximum likelihood.

        Parameters
        ----------
        during_analysis : bool
            Whether the model.results are being written during the analysis or after the non-linear search has finished.
        """

        results = []

        if hasattr(self, "evidence"):
            if self.evidence is not None:
                results += text_util.label_and_value_string(
                    label="Bayesian Evidence ",
                    value=self.evidence,
                    whitespace=90,
                    format_string="{:.8f}",
                )
                results += ["\n"]

        results += text_util.label_and_value_string(
            label="Maximum Likelihood ",
            value=self.maximum_log_likelihood,
            whitespace=90,
            format_string="{:.8f}",
        )
        results += ["\n\n"]

        results += ["Most Likely Model:\n\n"]
        most_likely = self.most_likely_vector

        formatter = text_formatter.TextFormatter()

        for i, prior_path in enumerate(self.model.unique_prior_paths):
            formatter.add((prior_path, self.format_str.format(most_likely[i])))
        results += [formatter.text + "\n"]

        if self.pdf_converged:

            results += self.results_from_sigma(sigma=3.0)
            results += ["\n"]
            results += self.results_from_sigma(sigma=1.0)

        else:

            results += [
                "\n WARNING: The chains have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += self.results_from_sigma(sigma=1.0)

        results += ["\n\ninstances\n"]

        formatter = text_formatter.TextFormatter()

        for t in self.model.path_float_tuples:
            formatter.add(t)

        results += ["\n" + formatter.text]

        text_util.output_list_of_strings_to_file(
            file=self.paths.file_results, list_of_strings=results
        )

    def output_pdf_plots(self):
        """Output plots of the probability density functions of the non-linear seach."""
        raise NotImplementedError()


class MCMCOutput(AbstractOutput):
    pass
