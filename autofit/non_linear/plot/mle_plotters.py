import logging

from autofit.non_linear.plot.samples_plotters import SamplesPlotter

from autofit.non_linear.plot.samples_plotters import skip_plot_in_test_mode

logger = logging.getLogger(__name__)

class MLEPlotter(SamplesPlotter):

    def subplot_parameters(self, use_log_y : bool = False, use_last_50_percent : bool = False, **kwargs):
        """
        Plots a subplot of every parameter against iteration number.

        The subplot extends over all free parameters in the model-fit, with the number of parameters per subplot
        given by the total number of free parameters in the model-fit.

        This often produces a large dynamic range in the y-axis. Plotting the y-axis on a log-scale or only
        plotting the last 50% of samples can make the plot easier to inspect.

        Parameters
        ----------
        use_log_y
            If True, the y-axis is plotted on a log-scale.
        use_last_50_percent
            If True, only the last 50% of samples are plotted.
        kwargs
            Additional key word arguments can be passed to the `plt.subplots` method.

        Returns
        -------

        """

        import matplotlib.pyplot as plt

        parameter_lists = self.samples.parameters_extract

        plt.subplots(self.model.total_free_parameters, 1, figsize=(12, 3 * len(parameter_lists)))

        for i, parameters in enumerate(parameter_lists):

            iteration_list = range(len(parameter_lists[0]))

            plt.subplot(self.model.total_free_parameters, 1, i + 1)

            if use_last_50_percent:

                iteration_list = iteration_list[int(len(iteration_list) / 2) :]
                parameters = parameters[int(len(parameters) / 2) :]

            if use_log_y:
                plt.semilogy(iteration_list, parameters, c="k")
            else:
                plt.plot(iteration_list, parameters, c="k")

            plt.xlabel("Iteration", fontsize=16)
            plt.ylabel(self.model.parameter_labels_with_superscripts_latex[i], fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

        filename = "subplot_parameters"

        if use_log_y:
            filename += "_log_y"

        if use_last_50_percent:
            filename += "_last_50_percent"

        self.output.subplot_to_figure(
            auto_filename=filename
        )
        plt.close()

    @skip_plot_in_test_mode
    def log_likelihood_vs_iteration(self, use_log_y : bool = False, use_last_50_percent : bool = False, **kwargs):
        """
        Plot the log likelihood of a model fit as a function of iteration number.

        For a maximum likelihood estimate, the log likelihood should increase with iteration number.

        This often produces a large dynamic range in the y-axis. Plotting the y-axis on a log-scale or only
        plotting the last 50% of samples can make the plot easier to inspect.

        Parameters
        ----------
        use_log_y
            If True, the y-axis is plotted on a log-scale.
        """

        import matplotlib.pyplot as plt

        log_likelihood_list = self.samples.log_likelihood_list
        iteration_list = range(len(log_likelihood_list))

        if use_last_50_percent:

            iteration_list = iteration_list[int(len(iteration_list) / 2) :]
            log_likelihood_list = log_likelihood_list[int(len(log_likelihood_list) / 2) :]

        plt.figure(figsize=(12, 12))

        if use_log_y:
            plt.semilogy(iteration_list, log_likelihood_list, c="k")
        else:
            plt.plot(iteration_list, log_likelihood_list, c="k")

        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Log Likelihood", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        title = "Log Likelihood vs Iteration"

        if use_log_y:

            title += " (Log Scale)"

        if use_last_50_percent:

            title += " (Last 50 Percent)"

        plt.title("Log Likelihood vs Iteration", fontsize=24)

        filename = "log_likelihood_vs_iteration"

        if use_log_y:
            filename += "_log_y"

        if use_last_50_percent:
            filename += "_last_50_percent"

        self.output.to_figure(
            auto_filename=filename,
        )
        plt.close()
