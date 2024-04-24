.. _database:

Database
========

The default behaviour of model-fitting results output is to be written to hard-disk in folders. These are simple to
navigate and manually check.

For small model-fitting tasks this is sufficient, however it does not scale well when performing many model fits to
large datasets, because manual inspection of results becomes time consuming.

All results can therefore be output to an sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database,
meaning that results can be loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation.
This database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can
be loaded.

This cookbook provides a concise reference to the database API.

**Contents:**

Ann overview of database functionality is given in the following sections:

- **Unique Identifiers**: How unique identifiers are used to ensure every entry of the database is unique.
- **Info**: Passing an ``info`` dictionary to the search to include information on the model-fit that is not part of the model-fit itself, which can be loaded via the database.
- **Building Database via Directory**: Build a database from results already written to hard-disk in an output folder.
- **Writing Directly To Database**: Set up a database session so results are written directly to the .sqlite database.
- **Files**: The files that are stored in the database that can be loaded and inspected.
- **Generators**: Why the database uses Python generators to load results.

The results that can be loaded via the database are described in the following sections:

- **Model**: The model fitted by the non-linear search.
- **Search**: The search used to perform the model-fit.
- **Samples**: The samples of the non-linear search (e.g. all parameter values, log likelihoods, etc.).
- **Samples Summary**: A summary of the samples of the non-linear search (e.g. the maximum log likelihood model) which can be faster to load than the full set of samples.
- **Info**: The ``info`` dictionary passed to the search.
- **Custom Output**: Extend ``Analysis`` classes to output additional information which can be loaded via the database (e.g. the data, maximum likelihood model data, etc.).

Using queries to load specific results is described in the following sections:

- **Querying Datasets**: Query based on the name of the dataset.
- **Querying Searches**: Query based on the name of the search.
- **Querying Models**: Query based on the model that is fitted.
- **Querying Results**: Query based on the results of the model-fit.
- **Querying Logic**: Use logic to combine queries to load specific results (e.g. AND, OR, etc.).

Unique Identifiers
------------------

Results output to hard-disk are contained in a folder named via a unique identifier (a
random collection of characters, e.g. ``8hds89fhndlsiuhnfiusdh``). The unique identifier changes if the model or
search change, to ensure different fits to not overwrite one another on hard-disk.

Each unique identifier is used to define every entry of the database as it is built. Unique identifiers therefore play
the same vital role for the database of ensuring that every set of results written to it are unique.

In this example, we fit 3 different datasets with the same search and model. Each ``dataset_name`` is therefore passed
in as the search's ``unique_tag`` to ensure 3 separate sets of results for each model-fit are written to the .sqlite
database.

Info
----

Information about the model-fit that is not part included in the model-fit itself can be made accessible via the
database by passing an ``info`` dictionary.

Below we write info on the dataset``s (hypothetical) data of observation and exposure time, which we will later show
the database can access.

For fits to large datasets this ensures that all relevant information for interpreting results is accessible.

.. code-block:: python

    info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

Results From Hard Disk
----------------------

We now perform a simple model-fit to 3 datasets, where the results are written to hard-disk using the standard
output directory structure and we will then build the database from these results. This behaviour is governed
by us inputting ``session=None``.

If you have existing results you wish to build a database for, you can therefore adapt this example you to do this.

Later in this example we show how results can also also be output directly to an .sqlite database, saving on hard-disk
space. This will be acheived by setting ``session`` to something that is not ``None``.

.. code-block:: python

    session = None

For each dataset we load it from hard-disc, set up a model and analysis and fit it with a non-linear search.

Note how the ``session`` is passed to the ``Dynesty`` search.

.. code-block:: python

    dataset_name_list = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

    model = af.Collection(gaussian=af.ex.Gaussian)

    model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.gaussian.sigma = af.GaussianPrior(
        mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
    )

    for dataset_name in dataset_name_list:
        dataset_path = path.join("dataset", "example_1d", dataset_name)

        data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
        noise_map = af.util.numpy_array_from_json(
            file_path=path.join(dataset_path, "noise_map.json")
        )

        analysis = af.ex.Analysis(data=data, noise_map=noise_map)

        search = af.DynestyStatic(
            name="database_example",
            path_prefix=path.join("features", "database"),
            unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
            session=session,  # This can instruct the search to write to the .sqlite database.
            nlive=50,
        )

        print(
            """
            The non-linear search has begun running.
            This Jupyter notebook cell with progress once search has completed, this could take a few minutes!
            """
        )

        result = search.fit(model=model, analysis=analysis, info=info)

    print("Search has finished run - you may now continue the notebook.")

Building Database via Directory
-------------------------------

The fits above wrote the results to hard-disk in folders, not as an .sqlite database file.

We build the database below, where the ``database_name`` corresponds to the name of your output folder and is also the
name of the ``.sqlite`` database file that is created.

If you are fitting a relatively small number of datasets (e.g. 10-100) having all results written to hard-disk (e.g.
for quick visual inspection) and using the database for sample wide analysis is beneficial.

We can optionally only include completed model-fits but setting ``completed_only=True``.

If you inspect the ``output`` folder, you will see a ``database.sqlite`` file which contains the results.

.. code-block:: python

    database_name = "database"

    agg = af.Aggregator.from_database(
       filename=f"{database_name}.sqlite", completed_only=False
    )

    agg.add_directory(directory=path.join("output", database_name)))

Writing Directly To Database
----------------------------

Results can be written directly to the .sqlite database file, skipping output to hard-disk entirely, by creating
a session and passing this to the non-linear search.

The code below shows how to do this.

This is ideal for tasks where model-fits to hundreds or thousands of datasets are performed, as it becomes unfeasible
to inspect the results of all fits on the hard-disk.

Our recommended workflow is to set up database analysis scripts using ~10 model-fits, and then scaling these up
to large samples by writing directly to the database.

.. code-block:: python

    session = af.db.open_database("database.sqlite")

    search = af.DynestyStatic(
        name="database_example",
        path_prefix=path.join("features", "database"),
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This can instruct the search to write to the .sqlite database.
        nlive=50,
    )


Files
-----

When performing fits which output results to hard-disc, a ``files`` folder is created containing .json / .csv files of
the model, samples, search, etc.

These are the files that are written to the database, and the aggregator load them via the database in order
to make them accessible in a Python script or Jupyter notebook.

Below, we will access these results using the aggregator's ``values`` method. A full list of what can be loaded is
as follows:

 - model: The ``model`` defined above and used in the model-fit (``model.json``).
 - search: The non-linear search settings of the fit (``search.json``).
 - samples: The non-linear search samples of the fit (``samples.csv``).
 - samples_summary: A summary of the samples results of the fit (``samples_summary.json``).
 - info: The info dictionary passed to the search (``info.json``).
 - covariance: The covariance matrix of the fit (``covariance.csv``).

The ``samples`` and ``samples_summary`` results contain a lot of repeated information. The ``samples`` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The ``samples_summary``
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the ``samples_summary`` is therefore a lot faster, as it does reperform calculations using the
full list of samples. Therefore, if the result you want is accessible via the ``samples_summary`` you should use it
but if not you can revert to the ``samples.

Generators
----------

Before using the aggregator to inspect results, lets discuss Python generators.

A generator is an object that iterates over a function when it is called. The aggregator creates all of the objects
that it loads from the database as generators (as opposed to a list, or dictionary, or another Python type).

This is because generators are memory efficient, as they do not store the entries of the database in memory
simultaneously. This contrasts objects like lists and dictionaries, which store all entries in memory all at once.
If you fit a large number of datasets, lists and dictionaries will use a lot of memory and could crash your computer!

Once we use a generator in the Python code, it cannot be used again. To perform the same task twice, the
generator must be remade it. This cookbook therefore rarely stores generators as variables and instead uses the
aggregator to create each generator at the point of use.

To create a generator of a specific set of results, we use the ``values`` method. This takes the ``name`` of the
object we want to create a generator of, for example inputting ``name=samples`` will return the results ``Samples``
object.

.. code-block:: python

    samples_gen = agg.values("samples")

By converting this generator to a list and printing it, it is a list of 3 ``SamplesNest`` objects, corresponding to
the 3 model-fits performed above.

.. code-block:: python

    print("Samples:\n")
    print(samples_gen)
    print("Total Samples Objects = ", len(agg), "\n")

Model
-----

The model used to perform the model fit for each of the 3 datasets can be loaded via the aggregator and printed.

.. code-block:: python

    model_gen = agg.values("model")

    for model in model_gen:
        print(model.info)

Search
------

The non-linear search used to perform the model fit can be loaded via the aggregator and printed.

.. code-block:: python

    search_gen = agg.values("search")

    for search in search_gen:
        print(search.info)

Samples
-------

The `Samples` class contains all information on the non-linear search samples, for example the value of every parameter
sampled using the fit or an instance of the maximum likelihood model.

The `Samples` class is described fully in the results cookbook.

.. code-block:: python

    for samples in agg.values("samples"):

        print("The tenth sample`s third parameter")
        print(samples.parameter_lists[9][2], "\n")

        instance = samples.max_log_likelihood()

        print("Max Log Likelihood `Gaussian` Instance:")
        print("Centre = ", instance.centre)
        print("Normalization = ", instance.normalization)
        print("Sigma = ", instance.sigma, "\n")

Samples Summary
---------------

The samples summary contains a subset of results access via the ``Samples``, for example the maximum likelihood model
and parameter error estimates.

Using the samples method above can be slow, as the quantities have to be computed from all non-linear search samples
(e.g. computing errors requires that all samples are marginalized over). This information is stored directly in the
samples summary and can therefore be accessed instantly.

.. code-block:: python

    for samples_summary in agg.values("samples_summary"):

        instance = samples_summary.max_log_likelihood()

        print("Max Log Likelihood `Gaussian` Instance:")
        print("Centre = ", instance.centre)
        print("Normalization = ", instance.normalization)
        print("Sigma = ", instance.sigma, "\n")

Info
----

The info dictionary passed to the search, discussed earlier in this cookbook, is accessible.

.. code-block:: python

    for info in agg.values("info"):
        print(info["date_of_observation"])
        print(info["exposure_time"])

The API for querying is fairly self explanatory. Through the combination of info based queries, model based
queries and result based queries a user has all the tools they need to fit extremely large datasets with many different
models and load only the results they are interested in for inspection and analysis.

Custom Output
-------------

The results accessible via the database (e.g. ``model``, ``samples``) are those contained in the ``files`` folder.

By extending an ``Analysis`` class with the methods ``save_attributes`` and ``save_results``,
custom files can be written to the ``files`` folder and become accessible via the database.

To save the objects in a human readable and loaded .json format, the `data` and `noise_map`, which are natively stored
as 1D numpy arrays, are converted to a suitable dictionary output format. This uses the **PyAutoConf** method
`to_dict`.

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

        def save_attributes(self, paths: af.DirectoryPaths):
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
            from autoconf.dictable import to_dict

            paths.save_json(name="data", object_dict=to_dict(self.data))
            paths.save_json(name="noise_map", object_dict=to_dict(self.noise_map))

        def save_results(self, paths: af.DirectoryPaths, result: af.Result):
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

            paths.save_json(name="model_data", object_dict=model_data)

Querying Datasets
-----------------

The aggregator can query the database, returning only specific fits of interested.

We can query using the ``dataset_name`` string we input into the model-fit above, in order to get the results
of a fit to a specific dataset.

For example, querying using the string ``gaussian_x1_1`` returns results for only the fit using the
second ``Gaussian`` dataset.

.. code-block:: python

    unique_tag = agg.search.unique_tag
    agg_query = agg.query(unique_tag == "gaussian_x1_1")

As expected, this list has only 1 ``SamplesNest`` corresponding to the second dataset.

.. code-block:: python

    print(agg_query.values("samples"))
    print("Total Samples Objects via dataset_name Query = ", len(agg_query), "\n")

If we query using an incorrect dataset name we get no results.

.. code-block:: python

    unique_tag = agg.search.unique_tag
    agg_query = agg.query(unique_tag == "incorrect_name")
    samples_gen = agg_query.values("samples")

Querying Searches
-----------------

We can query using the ``name`` of the non-linear search used to fit the model.

In this cookbook, all three fits used the same search, named ``database_example``. Query based on search name in this
example is therefore somewhat pointless.

However, querying based on the search name is useful for model-fits which use a range of searches, for example
if different non-linear searches are used multiple times.

As expected, the query using search name below contains all 3 results.

.. code-block:: python

    name = agg.search.name
    agg_query = agg.query(name == "database_example")

    print(agg_query.values("samples"))
    print("Total Samples Objects via name Query = ", len(agg_query), "\n")

Querying Models
---------------

We can query based on the model fitted.

For example, we can load all results which fitted a ``Gaussian`` model-component, which in this simple example is all
3 model-fits.

Querying via the model is useful for loading results after performing many model-fits with many different model
parameterizations to large (e.g. Bayesian model comparison).

[Note: the code ``agg.model.gaussian`` corresponds to the fact that in the ``Collection`` above, we named the model
component ``gaussian``. If this ``Collection`` had used a different name the code below would change
correspondingly. Models with multiple model components (e.g., ``gaussian`` and ``exponential``) are therefore also easily
accessed via the database.]

.. code-block:: python

    gaussian = agg.model.gaussian
    agg_query = agg.query(gaussian == af.ex.Gaussian)
    print("Total Samples Objects via `Gaussian` model query = ", len(agg_query), "\n")

Querying Results
----------------

We can query based on the results of the model-fit.

Below, we query the database to find all fits where the inferred value of ``sigma`` for the ``Gaussian`` is less
than 3.0 (which returns only the first of the three model-fits).

.. code-block:: python

    gaussian = agg.model.gaussian
    agg_query = agg.query(gaussian.sigma < 3.0)
    print("Total Samples Objects In Query `gaussian.sigma < 3.0` = ", len(agg_query), "\n")

Querying with Logic
-------------------

Advanced queries can be constructed using logic.

Below, we combine the two queries above to find all results which fitted a ``Gaussian`` AND (using the & symbol)
inferred a value of sigma less than 3.0.

The OR logical clause is also supported via the symbol |.

.. code-block:: python

    gaussian = agg.model.gaussian
    agg_query = agg.query((gaussian == af.ex.Gaussian) & (gaussian.sigma < 3.0))
    print(
        "Total Samples Objects In Query `Gaussian & sigma < 3.0` = ", len(agg_query), "\n"
    )

HowToFit
--------

The Database chapter of the **HowToFit** Jupyter notebooks give a full description of the database feature, including
examples of advanced queries and how to load and plot the results of a model-fit in more detail.
