import matplotlib
from autoconf import conf

import autofit as af
from src.dataset import dataset as ds
from src.fit import fit as f
from src.plot import dataset_plots, fit_plots

"""
The `Visualizer` is used by a phase to output results to the hard-disk. It is called both during the model-fit,
enabling the best-fit model of a `NonLinearSearch` to be output on-the-fly (e.g. whilst it is still running) and
at the end.
"""

"""
This allows the user to set the Matplotlib backend in the `config/visualize/general.ini` configuration file. The
benefit of this is that there can be issues with certain backends for different operating systems, which easier user
customization can make fixing easier.
"""

backend = conf.get_matplotlib_backend()

if backend not in "default":
    matplotlib.use(backend)

"""
The parent project **PyAutoConf** allows us to set up configuration files that a user can customize to control how
the model-fitting software behaves.

Checkout the configuration file `src/config/visualize/plots.ini`. Here, the user can turn on or off all the different
plots that are made during a model-fit. By default, we make it so that only subplots containing the data and fit are 
are output during the model-fit. However, the complete set of images are output at the end of the analysis.

Below, you can see we use the `conf.instance` function to load each entry from the config file. For example, the 
command `conf.instance["visualize"]["plots"]["data"]["plot_dataset_data] loads the corresponding entry in the config
file.
"""


class Visualizer:
    def __init__(self):

        self.plot_subplot_dataset = conf.instance["visualize"]["plots"]["dataset"][
            "subplot_dataset"
        ]
        self.plot_dataset_data = conf.instance["visualize"]["plots"]["dataset"]["data"]
        self.plot_dataset_noise_map = conf.instance["visualize"]["plots"]["dataset"][
            "noise_map"
        ]

        self.plot_subplot_fit = conf.instance["visualize"]["plots"]["fit"][
            "subplot_fit"
        ]
        self.plot_fit_data = conf.instance["visualize"]["plots"]["fit"]["data"]
        self.plot_fit_noise_map = conf.instance["visualize"]["plots"]["fit"][
            "noise_map"
        ]
        self.plot_fit_model_data = conf.instance["visualize"]["plots"]["fit"][
            "model_data"
        ]
        self.plot_fit_residual_map = conf.instance["visualize"]["plots"]["fit"][
            "residual_map"
        ]
        self.plot_fit_normalized_residual_map = conf.instance["visualize"]["plots"][
            "fit"
        ]["normalized_residual_map"]
        self.plot_fit_chi_squared_map = conf.instance["visualize"]["plots"]["fit"][
            "chi_squared_map"
        ]

    def visualize_dataset(self, paths: af.Paths, dataset: ds.Dataset):

        """For visualizing the dataset, we simply plot the plot methods here."""

        if self.plot_dataset_data:

            dataset_plots.data(
                dataset=dataset,
                output_filename="dataset_data",
                output_path=paths.image_path,
                output_format="png",
            )

        if self.plot_dataset_noise_map:

            dataset_plots.noise_map(
                dataset=dataset,
                output_filename="dataset_noise_map",
                output_path=paths.image_path,
                output_format="png",
            )

    def visualize_fit(self, paths: af.Paths, fit: f.FitDataset, during_analysis: bool):

        """
        If this function is called during an analysis, the during_analysis bool will be `True`. If there are
        images you only want to output at the end of the analysis, you can thus save them for this if clause only.

        For example, this phase only visualizes individual images of the fit`s normalized residual-map after the
        model fit has finished. For problems where there is a lot of visualization, and it can thus often take a
        long term to perform, limiting the output to after the analysis can be beneficial.
        """

        if self.plot_fit_data or not during_analysis:

            fit_plots.data(
                fit=fit,
                output_filename="fit_data",
                output_path=paths.image_path,
                output_format="png",
            )

        if self.plot_fit_noise_map or not during_analysis:

            fit_plots.noise_map(
                fit=fit,
                output_filename="fit_noise_map",
                output_path=paths.image_path,
                output_format="png",
            )

        if self.plot_fit_model_data or not during_analysis:

            fit_plots.model_data(
                fit=fit,
                output_filename="fit_model_data",
                output_path=paths.image_path,
                output_format="png",
            )

        if self.plot_fit_residual_map or not during_analysis:

            fit_plots.residual_map(
                fit=fit,
                output_filename="fit_residual_map",
                output_path=paths.image_path,
                output_format="png",
            )

        if self.plot_fit_normalized_residual_map or not during_analysis:

            fit_plots.normalized_residual_map(
                fit=fit,
                output_filename="fit_normalized_residual_map",
                output_path=paths.image_path,
                output_format="png",
            )

        if self.plot_fit_chi_squared_map or not during_analysis:

            fit_plots.chi_squared_map(
                fit=fit,
                output_filename="fit_chi_squared_map",
                output_path=paths.image_path,
                output_format="png",
            )
