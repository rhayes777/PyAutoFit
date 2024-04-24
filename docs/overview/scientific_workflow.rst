.. _scientific_workflow:

Scientific Workflow
===================

A scientific workflow are the tasks that you perform to undertake scientific study. This includes fitting models to
datasets, interpreting the results and gaining insight and intuition about your scientific problem.

Different problems require different scientific workflows, depending on the complexity of the model, the size of the
dataset and the computational run times of the analysis. For example, some problems have a single dataset, which
is fitted with many different models in order to gain scientific insight. Other problems have many thousands of datasets,
that are fitted with a single model, in order to perform a large scale scientific study.

The **PyAutoFit** API is flexible, customisable and extensible, enabling user to a develop scientific workflow
that is tailored to their specific problem.

This overview covers the key features of **PyAutoFit** that enable the development of effective scientific workflows,
which are as follows:

- **On The Fly**: Displaying results on-the-fly (e.g. in a Jupyter notebooks), providing immediate feedback for adapting a scientific workflow.
- **Hard Disk Output**: Output results to hard-disk with high levels of customization, enabling quick but detailed inspection of fits to many datasets.
- **Visualization**: Output model specific visualization to produce custom plots that further streamline result inspection.
- **Loading Results**: Load results from hard-disk to inspect and interpret the results of a model-fit.
- **Result Customization**: Customize the result returned by a fit to streamline scientific interpretation.
- **Model Composition**: Extensible model composition makes straight forward to fit many models with different parameterizations and assumptions.
- **Searches**: Support for many non-linear searches (nested sampling, MCMC etc.), so that a user finds the right method for their problem.
- **Configs**: Configuration files which set default model, fitting and visualization behaviour, streamlining model-fitting.
- **Multiple Datasets**: Dedicated support for simultaneously fitting multiple datasets, enabling scalable analysis of large datasets.
- **Database**: Store results in a relational sqlite3 database, enabling streamlined management of large modeling results.
- **Scaling Up**: Advise on how to scale up your scientific workflow from small datasets to large datasets.

On The Fly
----------

When a model-fit is running, information about the fit is displayed in user specified intervals on-the-fly.

The frequency of on-the-fly output is controlled by a search's ``iterations_per_update``, which specifies how often
this information is output. The example code below outputs on-the-fly information every 1000 iterations:

.. code-block:: python

    search = af.DynestyStatic(
        iterations_per_update=1000
    )

In a Jupyter notebook, the default behaviour is for this information to appear in the cell being run and for it to include:

- Text displaying the maximum likelihood model inferred so far and related information.
- A visual showing how the search has sampler parameter space so far, giving intuition on how the search is performing.

Here is an image of how this looks:

The most valuable on-the-fly output is often specific to the model and dataset you are fitting.

For example, it might be a ``matplotlib`` subplot showing the maximum likelihood model's fit to the dataset, complete
with residuals and other diagnostic information.

The on-the-fly output can be fully customized by extending the ``on_the_fly_output`` method of the ``Analysis`` class
being used to fit the model.

The example below shows how this is done for the simple example of fitting a 1D Gaussian profile:

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(self, data: np.ndarray, noise_map: np.ndarray):
            """
            Example Analysis class illustrating how to customize the on-the-fly output of a model-fit.
            """
            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def on_the_fly_output(self, instance):
            """
            During a model-fit, the `on_the_fly_output` method is called throughout the non-linear search.

            The `instance` passed into the method is maximum log likelihood solution obtained by the model-fit so far and it can be
            used to provide on-the-fly output showing how the model-fit is going.
            """
            xvalues = np.arange(analysis.data.shape[0])

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

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
            plt.show() # By using `plt.show()` the plot will be displayed in the Jupyter notebook.

Here is how the visuals appear in a Jupyter Notebook:

In the early stages of setting up a scientific workflow, on-the-fly output is invaluable.

It provides immediate feedback on how your model fitting is performing (which at the start of a project is often not very well!).
It also forces you to think first and foremost about how to visualize your fit and diagnose whether things are performing
well or not.

We recommend users starting a new model-fitting problem should always begin by setting up on-the-fly output!

.. note::

    The function ``on_the_fly_output`` is not implemented yet, we are working on this currently!

Hard Disk Output
----------------

By default, a non-linear search does not output its results to hard-disk and its results can only be inspected
in a Jupyter Notebook or Python script via the ``result`` that is returned.

However, the results of any non-linear search can be output to hard-disk by passing the ``name`` and / or ``path_prefix``
attributes, which are used to name files and output the results to a folder on your hard-disk.

The benefits of doing this include:

- Inspecting results via folders on your computer is more efficient than using a Jupyter Notebook for multiple datasets.
- Results are output on-the-fly, making it possible to check that a fit is progressing as expected mid way through.
- Additional information about a fit (e.g. visualization) can be output (see below).
- Unfinished runs can be resumed from where they left off if they are terminated.
- On high performance super computers results often must be output in this way.

The code below shows how to enable outputting of results to hard-disk:

.. code-block:: python

    search = af.Emcee(
        path_prefix=path.join("folder_0", "folder_1"),
        name="example_mcmc"
    )

The screenshot below shows the output folder where all output is enabled:

.. note::

    Screenshot needs to be added here.

Lets consider three parts of this output folder:

- **Unique Identifier**: Results are output to a folder which is a collection of random characters, which is uniquely generated based on the model fit. For scientific workflows where many models are fitted this means many fits an be performed without manually updating the output paths.
- **Info Files**: Files containing useful information about the fit are available, for example ``model.info`` contains the full model composition and ``search.summary`` contains information on how long the search has been running.
- **Files Folder**: The ``files`` folder contains detailed information about the fit, as ``.json`` files which can be loaded as (e.g. ``model.json`` can be used to load the``Model``), so that if you return to results at a later date you can remind yourself how the fit was performed.

**PyAutoFit** has lots more tools for customizing hard-disk output, for example configuration files controlling what gets output in order to manage hard-disk space use and model-specific ``.json`` files.

For many scientific workflows, being able to output so much information about each fit is integral to ensuring you inspect and interpret the results
accurately. On the other hand, there are many problems where outputting so much information to hard-disk may overwhelm a user and prohibit
scientific study, which is why it can be easily disabled by not passing the search a ``name`` or ``path prefix``!

Visualization
-------------

If search hard-disk output is enabled, visualization of the model-fit can also be output to hard-disk, which
for many scientific workflows is integral to assessing the quality of a fit quickly and effectively.

This is done by overwriting the ``Visualizer`` object of an ``Analysis`` class with a custom ``Visualizer`` class,
as illustrated below.

.. code-block:: python

     class Visualizer(af.Visualizer):

        @staticmethod
        def visualize_before_fit(
            analysis,
            paths: af.DirectoryPaths,
            model: af.AbstractPriorModel
        ):
            """
            Before a model-fit, the `visualize_before_fit` method is called to perform visualization.

            The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
            which is used to perform the visualization (e.g. it contains the data and noise map which are plotted).

            This can output visualization of quantities which do not change during the model-fit, for example the
            data and noise-map.

            The `paths` object contains the path to the folder where the visualization should be output, which is determined
            by the non-linear search `name` and other inputs.
            """

            import matplotlib.pyplot as plt

            xvalues = np.arange(analysis.data.shape[0])

            plt.errorbar(
                x=xvalues,
                y=analysis.data,
                yerr=analysis.noise_map,
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

        @staticmethod
        def visualize(
            analysis,
            paths: af.DirectoryPaths,
            instance,
            during_analysis
        ):
            """
            During a model-fit, the `visualize` method is called throughout the non-linear search.

            The function receives as input an instance of the `Analysis` class which is being used to perform the fit,
            which is used to perform the visualization (e.g. it generates the model data which is plotted).

            The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
            so far and it can be used to provide on-the-fly images showing how the model-fit is going.

            The `paths` object contains the path to the folder where the visualization should be output, which is determined
            by the non-linear search `name` and other inputs.
            """
            xvalues = np.arange(analysis.data.shape[0])

            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
            residual_map = analysis.data - model_data

            """
            The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`).
            """
            import matplotlib.pyplot as plt

            plt.errorbar(
                x=xvalues,
                y=analysis.data,
                yerr=analysis.noise_map,
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
                yerr=analysis.noise_map,
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

The ``Analysis`` class is defined following the same API as before, but now with its `Visualizer` class attribute
overwritten with the ``Visualizer`` class above.

.. code-block:: python

    class Analysis(af.Analysis):

        """
        This over-write means the `Visualizer` class is used for visualization throughout the model-fit.

        This `VisualizerExample` object is in the `autofit.example.visualize` module and is used to customize the
        plots output during the model-fit.

        It has been extended with visualize methods that output visuals specific to the fitting of `1D` data.
        """
        Visualizer = Visualizer

        def __init__(self, data, noise_map):
            """
            An Analysis class which illustrates visualization.
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

Visualization of the results of the search, such as the corner plot of what is called the "Probability Density
Function", are also automatically output during the model-fit on the fly.

Loading Results
---------------

Your scientific workflow will likely involve many model-fits, which will be output to hard-disk in folders.

A streamlined API for loading these results from hard-disk to Python variables, so they can be manipulated and
inspected in a Python script or Jupiter notebook is therefore essential.

The **PyAutoFit** aggregator provides this API, you simply point it at the folder containing the results and it
loads the results (and other information) of all model-fits in that folder.

.. code-block:: python

    from autofit.aggregator.aggregator import Aggregator

    agg = Aggregator.from_directory(
        directory=path.join("output", "result_folder"),
    )

The ``values`` method is used to specify the information that is loaded from the hard-disk, for example the
``samples`` of the model-fit.

The for loop below iterates over all results in the folder passed to the aggregator above.

.. code-block:: python

    for samples in agg.values("samples"):
        print(samples.parameter_lists[0])

Result loading uses Python generators to ensure that memory use is minimized, meaning that even when loading
thousands of results from hard-disk the memory use of your machine is not exceeded.

The `result cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_ gives a full run-through of
the tools that allow results to be loaded and inspected.

Result Customization
--------------------

The ``Result`` object is returned by a non-linear search after running the following code:

.. code-block:: python

    result = search.fit(model=model, analysis=analysis)

An effective scientific workflow ensures that this object contains all information a user needs to quickly inspect
the quality of a model-fit and undertake scientific interpretation.

The result can be can be customized to include additional information about the model-fit that is specific to your
model-fitting problem.

For example, for fitting 1D profiles, the ``Result`` could include the maximum log likelihood model 1D data:

.. code-block:: python

    print(result.max_log_likelihood_model_data_1d)

To do this we use the custom result API, where we first define a custom ``Result`` class which includes the
property ``max_log_likelihood_model_data_1d``:

.. code-block:: python

    class ResultExample(af.Result):

        @property
        def max_log_likelihood_model_data_1d(self) -> np.ndarray:
            """
            Returns the maximum log likelihood model's 1D model data.

            This is an example of how we can pass the `Analysis` class a custom `Result` object and extend this result
            object with new properties that are specific to the model-fit we are performing.
            """
            xvalues = np.arange(self.analysis.data.shape[0])

            return self.instance.model_data_1d_via_xvalues_from(instance=xvalues)

The custom result has access to the analysis class, meaning that we can use any of its methods or properties to
compute custom result properties.

To make it so that the ``ResultExample`` object above is returned by the search we overwrite the ``Result`` class attribute
of the ``Analysis`` and define a ``make_result`` object describing what we want it to contain:

.. code-block:: python

    class Analysis(af.Analysis):

        """
        This overwrite means the `ResultExample` class is returned after the model-fit.
        """
        Result = ResultExample

        def __init__(self, data, noise_map):
            """
            An Analysis class which illustrates custom results.
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

        def make_result(
            self,
            samples_summary: af.SamplesSummary,
            paths: af.AbstractPaths,
            samples: Optional[af.SamplesPDF] = None,
            search_internal: Optional[object] = None,
            analysis: Optional[object] = None,
        ) -> Result:
            """
            Returns the `Result` of the non-linear search after it is completed.

            The result type is defined as a class variable in the `Analysis` class (see top of code under the python code
            `class Analysis(af.Analysis)`.

            The result can be manually overwritten by a user to return a user-defined result object, which can be extended
            with additional methods and attribute specific to the model-fit.

            This example class does example this, whereby the analysis result has been overwritten with the `ResultExample`
            class, which contains a property `max_log_likelihood_model_data_1d` that returns the model data of the
            best-fit model. This API means you can customize your result object to include whatever attributes you want
            and therefore make a result object specific to your model-fit and model-fitting problem.

            The `Result` object you return can be customized to include:

            - The samples summary, which contains the maximum log likelihood instance and median PDF model.

            - The paths of the search, which are used for loading the samples and search internal below when a search
            is resumed.

            - The samples of the non-linear search (e.g. MCMC chains) also stored in `samples.csv`.

            - The non-linear search used for the fit in its internal representation, which is used for resuming a search
            and making bespoke visualization using the search's internal results.

            - The analysis used to fit the model (default disabled to save memory, but option may be useful for certain
            projects).

            Parameters
            ----------
            samples_summary
                The summary of the samples of the non-linear search, which include the maximum log likelihood instance and
                median PDF model.
            paths
                An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
            samples
                The samples of the non-linear search, for example the chains of an MCMC run.
            search_internal
                The internal representation of the non-linear search used to perform the model-fit.
            analysis
                The analysis used to fit the model.

            Returns
            -------
            Result
                The result of the non-linear search, which is defined as a class variable in the `Analysis` class.
            """
            return self.Result(
                samples_summary=samples_summary,
                paths=paths,
                samples=samples,
                search_internal=search_internal,
                analysis=self
            )

Result customization has full support for **latent variables**, which are parameters that are not sampled by the non-linear
search but are computed from the sampled parameters.

They are often integral to assessing and interpreting the results of a model-fit, as they present information
on the model in a different way to the sampled parameters.

The `result cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html>`_ gives a full run-through of
all the different ways the result can be customized.

Model Composition
-----------------

Many scientific workflows require composing and fitting many different models.

The simplest examples are when slight tweaks to the model are required, for example:

- **Parameter Assignment**: Fix certain parameters to input values or linking parameters in the model together so they have the same values.
- **Parameter Assertions**: Place assertions on model parameters, for example requiring that one parameter is higher than another parameter.
- **Model Arithmitic**: Use arithmitic to define relations between parameters, for example a ``y = mx + c`` where ``m`` and ``c`` are model parameters.

In more complex situations, models with many thousands of parameters consisting of many model components may be fitted.

**PyAutoFit**'s advanced model composition API has many tools for compositing complex models, including constructing
models from lists of Python classes and hierarchies of Python classes.

The `model cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html>`_ gives a full run-through of
the model composition API.

Searches
--------

Different model-fitting problems require different methods to fit the model.

The search appropriate for your problem depends on many factors:

- **Model Dimensions**: How many parameters does the model and its non-linear parameter space consist of?
- **Model Complexity**: Different models have different parameter degeneracies requiring different non-linear search techniques.
- **Run Times**: How fast does it take to evaluate a likelihood and perform the model-fit?

**PyAutoFit** supports many non-linear searches ensuring that each user can find the best method for their problem.

In the early stages of setting up your scientific workflow, you should experiment with different searches, determine which
reliable infer the maximum likelihood fits to the data and profile which ones do so in the faster times.

The `search cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html>`_ gives a full run-through of
all non-linear searches that are available and how to customize them.

.. note::

    There are currently no documentation guiding reads on what search might be appropriate for their problem and how to
    profile and experiment with different methods. Writing such documentation is on the to do list and will appear
    in the future. However, you can make progress now simply using visuals output by PyAutoFit and the ``search.summary` file.

Configs
-------

As you develop your scientific workflow, you will likely find that you are often setting up models with the same priors
every time and using the same non-linear search settings.

This can lead to long lines of Python code repeating the same inputs.

Configuration files can be set up to input default values, meaning that the same prior inputs and settings do not need
to be repeated in every script. This produces more concise Python code and means you have less to think about when
performing model-fitting.

The `configs cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/configs.html>`_ gives a full run-through of
configuration file setup.

Multiple Datasets
-----------------

Many model-fitting problems require multiple datasets to be fitted simultaneously in order to provide the best constraints on the model.

**PyAutoFit** makes it straight forward to scale-up your scientific workflow to fits to multiple datasets. All you
do is define ``Analysis`` classes describing how the model fits each dataset and sum them together:

.. code-block:: python

    analysis_0 = Analysis(data=data_0, noise_map=noise_map_0)
    analysis_1 = Analysis(data=data_1, noise_map=noise_map_1)

    # This means the model is fitted to both datasets simultaneously.

    analysis = analysis_0 + analysis_1

    # summing a list of analysis objects is also a valid API:

    analysis = sum([analysis_0, analysis_1])

By summing analysis objects the following happen:

- The log likelihood values computed by the ``log_likelihood_function`` of each individual analysis class are summed to give an overall log likelihood value that the non-linear search samples when model-fitting.

- The output path structure of the results goes to a single folder, which includes sub-folders for the visualization of every individual analysis object based on the ``Analysis`` object's ``visualize`` method.

In the example above, the same ``Analysis`` class was used twice (to set up ``analysis_0`` and ``analysis_1``) and summed.

**PyAutoFit** supports the summing together of different ``Analysis`` classes, which may take as input completely different
datasets and fit the model to them (via the ``log_likelihood_function``) following a completely different procedure.

The `multiple datasets cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/multiple_datasets.html>`_ gives a full run-through
of fitting multiple dataset. This includes a dedicated API for customizing how the model changes over the different datasets
and how the result return becomes a list containing information on the fit to every dataset.

Database
--------

The default behaviour of model-fitting results output is to be written to hard-disk in folders. These are simple to
navigate and manually check.

For small scientific workflows and model-fitting tasks this is sufficient, however it does not scale well when
performing many model fits to large datasets, because manual inspection of results becomes time consuming.

All results can therefore be output to an sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database,
meaning that results can be loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation.
This database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can
be loaded.

The `database cookbook <https://pyautofit.readthedocs.io/en/latest/cookbooks/multiple_datasets.html>`_ gives a full run-through
of how to use the database functionality.

Scaling Up
----------

Irrespective of your final scientific goal, you should always

Initially, the study will be performed on a small number of datasets (e.g. ~10s of datasets), as the user develops
their model and gains insight into what works well. This is a manual process of trial and error, which often involves
fitting many different models to the datasets and inspecting the result to gain insight on what models are good.
Their scientific workflow must be flexible enough to allow them to quickly fit many different models to their data,
and output the results in a format that is quick and easy to inspect and interpret.

Eventually, one may then scale up to a large number of datasets (e.g. ~1000s of datasets). Manual inspection of
individual results becomes infeasible, and the scientific workflow requires a more automated apporach to model fitting
and interpretation. This may also see the analysis move to a high performance computing, meaning that result output
must be suitable for this environment.

**PyAutoFit** enables the development of effective scientific workflows for both small and large datasets, thanks
to the following features:


**PyAutoFit** supports **on-the-fly** output whilst the model-fit is running.

For example, in a Jupyter notebook, text displaying the maximum likelihood model inferred so far alongside visuals
showing the parameter sampling are output to the cell as the search runs. On the fly output can be fully customized
with visualization specific to your model and data, as shown in the following overviews and cookbooks.