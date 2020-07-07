from howtofit.chapter_1_introduction.tutorial_5_visualization_masking.src.plot import (
    dataset_plots,
    fit_plots,
)

"""
The visualizer is used by a phase to output results to the hard-disk. It is called both during the model-fit,
enabling the best-fit model of a non-linear search to be output on-the-fly (e.g. whilst it is still running) and
at the end.

The 'image_path' specifies the path where images are output. By default, this is the image_path of the search,
so the folder 'output/phase_name/image'.
"""


class AbstractVisualizer:
    def __init__(self, image_path):

        self.image_path = image_path


class Visualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):

        """When the Visualizer is instantiated, the masked dataset is passed to it and visualized."""

        super().__init__(image_path)

        self.masked_dataset = masked_dataset

        """For visualizing the dataset, we simmply plot the plot methods here."""

        dataset_plots.data(
            dataset=masked_dataset,
            output_filename="dataset_data",
            output_path=self.image_path,
            output_format="png",
        )
        dataset_plots.noise_map(
            dataset=masked_dataset,
            output_filename="dataset_noise_map",
            output_path=self.image_path,
            output_format="png",
        )

    def visualize_fit(self, fit, during_analysis):

        """
        The fit is visualized during the model-fit, thus it requires its own method which is called by the non-linear
        search every set number of intervals.

        Unlike the dataset plots above, the fit 'data' and 'noise-map' are masked.
        """

        fit_plots.data(
            fit=fit,
            output_filename="fit_data",
            output_path=self.image_path,
            output_format="png",
        )
        fit_plots.noise_map(
            fit=fit,
            output_filename="fit_noise_map",
            output_path=self.image_path,
            output_format="png",
        )
        fit_plots.model_data(
            fit=fit,
            output_filename="fit_model_data",
            output_path=self.image_path,
            output_format="png",
        )
        fit_plots.residual_map(
            fit=fit,
            output_filename="fit_residual_map",
            output_path=self.image_path,
            output_format="png",
        )
        fit_plots.chi_squared_map(
            fit=fit,
            output_filename="fit_chi_squared_map",
            output_path=self.image_path,
            output_format="png",
        )

        if not during_analysis:

            """
            If this function is called during an analysis, the during_analysis bool will be 'True'. If there are
            images you only want to output at the end of the analysis, you can thus save them for this if clause only.

            For example, this phase only visualizes individual images of the fit's normalized residual-map after the
            model fit has finished. For problems where there is a lot of visualization, and it can thus often take a
            long term to perform, limiting the output to after the analysis can be beneficial.
            """

            fit_plots.normalized_residual_map(
                fit=fit,
                output_filename="fit_normalized_residual_map",
                output_path=self.image_path,
                output_format="png",
            )
