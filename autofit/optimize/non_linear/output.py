import os

from autofit import conf
from autofit.tools import text_formatter, text_util
import numpy as np


class AbstractOutput(object):
    def __init__(self, model, paths):
        self.model = model
        self.paths = paths

    @property
    def maximum_log_likelihood(self):
        raise NotImplementedError()

    @property
    def most_probable_model_parameters(self):
        raise NotImplementedError()

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        raise NotImplementedError()

    @property
    def most_probable_model_instance(self):
        return self.model.instance_from_physical_vector(
            physical_vector=self.most_probable_model_parameters
        )

    @property
    def most_likely_model_instance(self):
        return self.model.instance_from_physical_vector(
            physical_vector=self.most_likely_model_parameters
        )

    def model_parameters_at_sigma_limit(self, sigma_limit):
        raise NotImplementedError()

    def model_parameters_at_upper_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(
            map(
                lambda param: param[1],
                self.model_parameters_at_sigma_limit(sigma_limit),
            )
        )

    def model_parameters_at_lower_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(
            map(
                lambda param: param[0],
                self.model_parameters_at_sigma_limit(sigma_limit),
            )
        )

    def model_errors_at_sigma_limit(self, sigma_limit):
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def model_errors_at_upper_sigma_limit(self, sigma_limit):
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        return list(
            map(
                lambda upper, most_probable: upper - most_probable,
                uppers,
                self.most_probable_model_parameters,
            )
        )

    def model_errors_at_lower_sigma_limit(self, sigma_limit):
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)
        return list(
            map(
                lambda lower, most_probable: most_probable - lower,
                lowers,
                self.most_probable_model_parameters,
            )
        )

    def model_errors_instance_at_sigma_limit(self, sigma_limit):
        return self.model.instance_from_physical_vector(
            physical_vector=self.model_errors_at_sigma_limit(sigma_limit=sigma_limit)
        )

    def model_errors_instance_at_upper_sigma_limit(self, sigma_limit):
        return self.model.instance_from_physical_vector(
            physical_vector=self.model_errors_at_upper_sigma_limit(
                sigma_limit=sigma_limit
            )
        )

    def model_errors_instance_at_lower_sigma_limit(self, sigma_limit):
        return self.model.instance_from_physical_vector(
            physical_vector=self.model_errors_at_lower_sigma_limit(
                sigma_limit=sigma_limit
            )
        )

    def gaussian_priors_at_sigma_limit(self, sigma_limit):
        """Compute the Gaussian Priors these results should be initialzed with in the next phase, by taking their \
        most probable values (e.g the means of their PDF) and computing the error at an input sigma_limit.

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.most_probable_model_parameters
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)

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

    @property
    def total_samples(self):
        raise NotImplementedError()

    def sample_likelihood_from_sample_index(self, sample_index):
        raise NotImplementedError()

    def sample_model_parameters_from_sample_index(self, sample_index):
        raise NotImplementedError()

    def sample_model_instance_from_sample_index(self, sample_index):
        """Setup a model instance of a weighted sample.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        model_parameters = self.sample_model_parameters_from_sample_index(
            sample_index=sample_index
        )

        return self.model.instance_from_physical_vector(
            physical_vector=model_parameters
        )

    def values_offset_from_input_model_parameters(self, input_model_parameters):
        return list(
            map(
                lambda input, most_probable: most_probable - input,
                input_model_parameters,
                self.most_probable_model_parameters,
            )
        )

    def results_from_sigma_limit(self, limit):

        lower_limits = self.model_parameters_at_lower_sigma_limit(sigma_limit=limit)
        upper_limits = self.model_parameters_at_upper_sigma_limit(sigma_limit=limit)

        sigma_formatter = text_formatter.TextFormatter()

        for i, prior_path in enumerate(self.model.unique_prior_paths):
            value = self.format_str.format(self.most_probable_model_parameters[i])
            upper_limit = self.format_str.format(upper_limits[i])
            lower_limit = self.format_str.format(lower_limits[i])
            value = value + " (" + lower_limit + ", " + upper_limit + ")"
            sigma_formatter.add((prior_path, value))

        return "\n\nMost probable model ({} sigma limits):\n\n{}".format(
            limit, sigma_formatter.text
        )

    @property
    def param_labels(self):
        """The param_names vector is a list each parameter's analysis_path, and is used for *GetDist* visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""

        paramnames_labels = []
        prior_class_dict = self.model.prior_class_dict
        prior_prior_model_dict = self.model.prior_prior_model_dict

        for prior_name, prior in self.model.prior_tuples_ordered_by_id:
            param_string = conf.instance.label.label(prior_name)
            prior_model = prior_prior_model_dict[prior]
            cls = prior_class_dict[prior]
            cls_string = "{}{}".format(
                conf.instance.label.subscript(cls), prior_model.component_number + 1
            )
            param_label = "{}_{{\\mathrm{{{}}}}}".format(param_string, cls_string)
            paramnames_labels.append(param_label)

        return paramnames_labels

    def create_paramnames_file(self):
        """The param_names file lists every parameter's analysis_path and Latex tag, and is used for *GetDist*
        visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""
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

        try:
            os.makedirs(self.paths.backup_path)
        except FileExistsError:
            pass

        self.create_paramnames_file()

        text_util.output_list_of_strings_to_file(
            file=self.paths.file_model_info, list_of_strings=self.model.info
        )

    def latex_results_at_sigma_limit(self, sigma_limit, format_str="{:.2f}"):

        labels = self.param_labels
        most_probables = self.most_probable_model_parameters
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)

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
    def format_str(self):
        decimal_places = conf.instance.general.get(
            "output", "model_results_decimal_places", int
        )
        return "{:." + str(decimal_places) + "f}"

    @property
    def pdf(self):
        raise NotImplementedError()

    @property
    def pdf_converged(self):
        raise NotImplementedError()

    def output_results(self, during_analysis):

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
        most_likely = self.most_likely_model_parameters

        formatter = text_formatter.TextFormatter()

        for i, prior_path in enumerate(self.model.unique_prior_paths):
            formatter.add((prior_path, self.format_str.format(most_likely[i])))
        results += [formatter.text + "\n"]

        if self.pdf_converged:

            results += self.results_from_sigma_limit(limit=3.0)
            results += ["\n"]
            results += self.results_from_sigma_limit(limit=1.0)

        else:

            results += [
                "\n WARNING: The chains have not converged enough to compute a PDF and model errors. \n "
                "The model below over estimates errors. \n\n"
            ]
            results += self.results_from_sigma_limit(limit=1.0)

        results += ["\n\ninstances\n"]

        formatter = text_formatter.TextFormatter()

        for t in self.model.path_float_tuples:
            formatter.add(t)

        results += ["\n" + formatter.text]

        text_util.output_list_of_strings_to_file(
            file=self.paths.file_results, list_of_strings=results
        )

    def output_pdf_plots(self):
        raise NotImplementedError()


class MCMCOutput(AbstractOutput):

    pass


class NestedSamplingOutput(AbstractOutput):
    def sample_weight_from_sample_index(self, sample_index):
        raise NotImplementedError()

    @property
    def evidence(self):
        raise NotImplementedError()

    def output_pdf_plots(self):

        import getdist.plots
        import matplotlib

        backend = conf.instance.visualize.get("figures", "backend", str)
        matplotlib.use(backend)
        import matplotlib.pyplot as plt

        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.visualize.get(
            "plots", "plot_pdf_1d_params", bool
        )

        if plot_pdf_1d_params:

            for param_name in self.model.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(
                    fname="{}/pdf_{}_1D.png".format(self.paths.pdf_path, param_name)
                )

        plt.close()

        plot_pdf_triangle = conf.instance.visualize.get(
            "plots", "plot_pdf_triangle", bool
        )

        if plot_pdf_triangle:

            try:
                pdf_plot.triangle_plot(roots=self.pdf)
                pdf_plot.export(fname="{}/pdf_triangle.png".format(self.paths.pdf_path))
            except Exception as e:
                print(type(e))
                print(
                    "The PDF triangle of this non-linear search could not be plotted. This is most likely due to a "
                    "lack of smoothness in the sampling of parameter space. Sampler further by decreasing the "
                    "parameter evidence_tolerance."
                )

        plt.close()
