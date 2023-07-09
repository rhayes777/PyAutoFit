.. _analysis:

Analysis
========

The ``Analysis`` class is the interface between the data and model, whereby a ``log_likelihood_function`` is defined
and called by the non-linear search fitting to fit the model.

This cookbook provides an overview of how to use and extend ``Analysis`` objects in **PyAutoFit**.

**Contents:**

- Example: A simple example of an analysis class which can be adapted for you use-case.
- Customization: Customizing an analysis class with different data inputs and editing the ``log_likelihood_function``.
- Visualization: Adding a ``visualize`` method to the analysis so that model-specific visuals are output to hard-disk.
- Custom Output: Add methods which output model-specific results to hard-disk in the ``files`` folder (e.g. as .json files) to aid in the interpretation of results.

Example
-------

An example simple ``Analysis`` class, to remind ourselves of the basic structure and inputs.

This can be adapted for your use case.

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(self, data: np.ndarray, noise_map: np.ndarray):
            """
            The `Analysis` class acts as an interface between the data and model in **PyAutoFit**.

            Its `log_likelihood_function` defines how the model is fitted to the data and it is called many times by
            the non-linear search fitting algorithm.

            In this example the `Analysis` `__init__` constructor only contains the `data` and `noise-map`, but it can be
            easily extended to include other quantities.

            Parameters
            ----------
            data
                A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
            noise_map
                A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
                metric, the log likelihood.
            """
            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance) -> float:
            """
            Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

            The data is fitted using an `instance` of the `Gaussian` class where its `model_data_1d_via_xvalues_from`
            is called in order to create a model data representation of the Gaussian that is fitted to the data.
            """

            xvalues = np.arange(self.data.shape[0])

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            chi_squared = sum(chi_squared_map)
            noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map ** 2.0))
            log_likelihood = -0.5 * (chi_squared + noise_normalization)

            return log_likelihood

An instance of the analysis class is created as follows.

.. code-block:: python

    dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    analysis = Analysis(data=data, noise_map=noise_map)

Customization
-------------

The ``Analysis`` class can be fully customized to be suitable for your model-fit.

For example, additional inputs can be included in the ``__init__`` constructor and used in the ``log_likelihood_function``.
if they are required for your ``log_likelihood_function`` to work.

The example below includes three additional inputs:

 - Instead of inputting a ``noise_map``, a ``noise_covariance_matrix`` is input, which means that corrrlated noise is
   accounted for in the ``log_likelihood_function``.

 - A ``mask`` is input which masks the data such that certain data points are omitted from the log likelihood

 - A ``kernel`` is input which can account for certain blurring operations during data acquisition.

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(
                self,
                data: np.ndarray,
                noise_covariance_matrix: np.ndarray,
                mask: np.ndarray,
                kernel: np.ndarray
        ):
            """
            The `Analysis` class which has had its inputs edited for a different model-fit.

            Parameters
            ----------
            data
                A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
            noise_covariance_matrix
                A 2D numpy array containing the noise values and their covariances for the data, used for computing the
                goodness of fit whilst accounting for correlated noise.
            mask
                A 1D numpy array containing a mask, where `True` values mean a data point is masked and is omitted from
                the log likelihood.
            kernel
                A 1D numpy array containing the blurring kernel of the data, used for creating the model data.
            """
            super().__init__()

            self.data = data
            self.noise_covariance_matrix = noise_covariance_matrix
            self.mask = mask
            self.kernel = kernel

        def log_likelihood_function(self, instance) -> float:
            """
            The `log_likelihood_function` now has access to the  `noise_covariance_matrix`, `mask` and `kernel`
            input above.
            """
            print(self.noise_covariance_matrix)
            print(self.mask)
            print(self.kernel)

            """
            We do not provide a specific example of how to use these inputs in the `log_likelihood_function` as they are
            specific to your model fitting problem.

            The key point is that any inputs required to compute the log likelihood can be passed into the `__init__`
            constructor of the `Analysis` class and used in the `log_likelihood_function`.
            """

            log_likelihood = None

            return log_likelihood

An instance of the analysis class is created as follows.

.. code-block:: python

    dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))

    noise_covariance_matrix = np.ones(shape=(data.shape[0], data.shape[0]))
    mask = np.full(fill_value=False, shape=data.shape)
    kernel = np.full(fill_value=1.0, shape=data.shape)

    analysis = Analysis(
        data=data, noise_covariance_matrix=noise_covariance_matrix, mask=mask, kernel=kernel
    )

Visualization
-------------

If a ``name`` is input into a non-linear search, all results are output to hard-disk in a folder.

By extending the ``Analysis`` class with a ``visualize_before_fit`` and / or ``visualize`` function, model specific
visualization will also be output into an ``image`` folder, for example as ``.png`` files.

This uses the maximum log likelihood model of the model-fit inferred so far.

Visualization of the results of the search, such as the corner plot of what is called the "Probability Density
Function", are also automatically output during the model-fit on the fly.

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(self, data, noise_map):
            """
            We use the simpler Analysis class above for this example.
            """
            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):
            """
            The `log_likelihood_function` is identical to the example above
            """
            xvalues = np.arange(self.data.shape[0])

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            chi_squared = sum(chi_squared_map)
            noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
            log_likelihood = -0.5 * (chi_squared + noise_normalization)

            return log_likelihood

        def visualize_before_fit(
            self, paths: af.DirectoryPaths, model: af.AbstractPriorModel
        ):
            """
            Before a model-fit, the `visualize_before_fit` method is called to perform visualization.

            This can output visualization of quantities which do not change during the model-fit, for example the
            data and noise-map.

            The `paths` object contains the path to the folder where the visualization should be output, which is determined
            by the non-linear search `name` and other inputs.
            """

            import matplotlib.pyplot as plt

            xvalues = np.arange(self.data.shape[0])

            plt.errorbar(
                x=xvalues,
                y=self.data,
                yerr=self.noise_map,
                color="k",
                ecolor="k",
                elinewidth=1,
                capsize=2,
            )
            plt.title("Maximum Likelihood Fit")
            plt.xlabel("x value of profile")
            plt.ylabel("Profile Normalization")
            plt.savefig(path.join(paths.image_path, f"data.png"))
            plt.clf()

        def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):
            """
            During a model-fit, the `visualize` method is called throughout the non-linear search.

            The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
            so far and it can be used to provide on-the-fly images showing how the model-fit is going.

            The `paths` object contains the path to the folder where the visualization should be output, which is determined
            by the non-linear search `name` and other inputs.
            """
            xvalues = np.arange(self.data.shape[0])

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
            residual_map = self.data - model_data

            """
            The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
            """
            import matplotlib.pyplot as plt

            plt.errorbar(
                x=xvalues,
                y=self.data,
                yerr=self.noise_map,
                color="k",
                ecolor="k",
                elinewidth=1,
                capsize=2,
            )
            plt.plot(xvalues, model_data, color="r")
            plt.title("Maximum Likelihood Fit")
            plt.xlabel("x value of profile")
            plt.ylabel("Profile Normalization")
            plt.savefig(path.join(paths.image_path, f"model_fit.png"))
            plt.clf()

            plt.errorbar(
                x=xvalues,
                y=residual_map,
                yerr=self.noise_map,
                color="k",
                ecolor="k",
                elinewidth=1,
                capsize=2,
            )
            plt.title("Residuals of Maximum Likelihood Fit")
            plt.xlabel("x value of profile")
            plt.ylabel("Residual")
            plt.savefig(path.join(paths.image_path, f"model_fit.png"))
            plt.clf()

Custom Output
-------------

When performing fits which output results to hard-disc, a ``files`` folder is created containing .json / .csv files of
the model, samples, search, etc.

These files are human readable and help one quickly inspect and interpret results.

By extending an ``Analysis`` class with the methods ``save_attributes_for_aggregator`` and ``save_results_for_aggregator``,
custom files can be written to the ``files`` folder to further aid this inspection.

These files can then also be loaded via the database, as described in the database cookbook.

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(self, data: np.ndarray, noise_map: np.ndarray):
            """
            Standard Analysis class example used throughout PyAutoFit examples.
            """
            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance) -> float:
            """
            Standard log likelihood function used throughout PyAutoFit examples.
            """

            xvalues = np.arange(self.data.shape[0])

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            chi_squared = sum(chi_squared_map)
            noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))
            log_likelihood = -0.5 * (chi_squared + noise_normalization)

            return log_likelihood

        def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):
            """
            Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
            folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

            For this analysis, it uses the `AnalysisDataset` object's method to output the following:

            - The dataset's data as a .json file.
            - The dataset's noise-map as a .json file.

            These are accessed using the aggregator via `agg.values("data")` and `agg.values("noise_map")`.

            Parameters
            ----------
            paths
                The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
                visualization, and the pickled objects used by the aggregator output by this function.
            """
            # The path where data.json is saved, e.g. output/dataset_name/unique_id/files/data.json

            file_path = (path.join(paths._json_path, "data.json"),)

            with open(file_path, "w+") as f:
                json.dump(self.data, f, indent=4)

            # The path where noise_map.json is saved, e.g. output/noise_mapset_name/unique_id/files/noise_map.json

            file_path = (path.join(paths._json_path, "noise_map.json"),)

            with open(file_path, "w+") as f:
                json.dump(self.noise_map, f, indent=4)

        def save_results_for_aggregator(self, paths: af.DirectoryPaths, result: af.Result):
            """
            At the end of a model-fit,  this routine saves attributes of the `Analysis` object to the `files`
            folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

            For this analysis it outputs the following:

            - The maximum log likelihood model data as a .json file.

            This is accessed using the aggregator via `agg.values("model_data")`.

            Parameters
            ----------
            paths
                The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
                visualization and the pickled objects used by the aggregator output by this function.
            result
                The result of a model fit, including the non-linear search, samples and maximum likelihood model.
            """
            xvalues = np.arange(self.data.shape[0])

            instance = result.max_log_likelihood_instance

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

            # The path where model_data.json is saved, e.g. output/dataset_name/unique_id/files/model_data.json

            file_path = (path.join(paths._json_path, "model_data.json"),)

            with open(file_path, "w+") as f:
                json.dump(model_data, f, indent=4)