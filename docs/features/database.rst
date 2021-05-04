.. _database:

Database
--------

The default behaviour of **PyAutoFit** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check the model-fitting results. For small model-fitting tasks this is
sufficient, however many users have a need to perform many model fits to very large datasets, making the manual
inspection of results time consuming.

PyAutoFit's database feature outputs all model-fitting results as a sqlite3 (https://docs.python.org/3/library/sqlite3.html)
relational database, such that all results can be efficiently loaded into a Jupyter notebook or Python script for
inspection, analysis and interpretation. This database supports advanced querying, so that specific
model-fits (e.g., which fit a certain model or dataset) can be loaded.

To make it so that results are output to an .sqlite database we simply open a database session and pass this session
to the non-linear search:

.. code-block:: bash

    session = af.db.open_database("database.sqlite")

    emcee = af.Emcee(
        path_prefix=path.join("features", "database"),
        session=session,  # This instructs the search to write to the .sqlite database.
    )

When a model-fit is performed, a unique identifier is generated based on the model and non-linear search. However,
if we were to fit many different datasets with the same model and non-linear search, they would all use the same
unique identifier and not be distinguishable by the database.

We can overcome this by using the name of the dataset as the ``unique_tag`` passed to the search, which is used alongside
the model and search to create the unique identifier:

.. code-block:: bash

    session = af.db.open_database("database.sqlite")

    dataset_name = "example_dataset_0"

    emcee = af.Emcee(
        path_prefix=path.join("features", "database"),
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This instructs the search to write to the .sqlite database.
    )

Lets suppose that we have performed 100 model-fits to 100 1D Gaussians, and when we ran **PyAutoFit** we told it
to write to the ``.sqlite`` database file. We can load these results in a Python script or Jupyter notebook using
the ``Aggregator``:

.. code-block:: bash

    agg = Aggregator.from_database("path/to/output/database.sqlite")

We can now use the ``Aggregator`` to inspect the results of all model-fits. For example, we can load the ``Samples``
object of all 100 model-fits, which contains information on the best-fit model, posterior, Bayesian evidence, etc.

Below, we use the samples generator to create a list of the maximum log likelihood of every model-fit and print it:

.. code-block:: bash

    for samples in agg.values("samples"):

        print(max(samples.log_likelihood))

This object (and all objects loaded by the ``Aggregator``) are returned as a generator (as opposed to a list,
dictionary or other Python type). This is because generators do not store large arrays or classes in memory until they
are used, ensuring that when we are manipulating large sets of results we do not run out of memory!

We can iterate over the samples to print the maximum log likelihood model ``Gaussian`` of every fit:

.. code-block:: bash

    for samps in agg.values("samples"):

        instance = samps.max_log_likelihood_instance

        print("Maximum Likelihood Model-fit \n")
        print("centre = ", instance.centre)
        print("intensity = ", instance.intensity)
        print("sigma = ", instance.sigma)


The ``Aggregator`` contains tools for querying the database for certain results, for example to load subsets of
model-fits. This can be done in many different ways, depending on what information you want.

Below, we query based on the model fitted. For example, we can load all results which fitted a ``Gaussian``
model-component, which in this simple example is all 100 model-fits (note that when we performed the model fit, we
composed model using the name ``gaussian``):

.. code-block:: bash

    gaussian = agg.gaussian
    agg_query = agg.query(gaussian == m.Gaussian)

Queries using the results of model-fitting are also supported. Below, we query the database to find all fits where the
inferred value of ``sigma`` for the ``Gaussian`` is less than 3.0:

.. code-block:: bash

    agg_query = agg.query(gaussian.sigma < 3.0)

Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted a ``Gaussian`` AND (using the & symbol) inferred a value of sigma less than 3.0.

The OR logical clause is also supported via the symbol |.

.. code-block:: bash

    agg_query = agg.query((gaussian == m.Gaussian) & (gaussian.sigma < 3.0))

We can query using the ``unique_tag`` to load the model-fit to a specific dataset:

.. code-block:: bash

    agg_query = agg.query(agg.unique_tag == "example_dataset_0")

An ``info`` dictionary can be passed into a model-fit, which contains information on the model-fit. The example below
creates an ``info`` dictionary which is passed to the model-fit, which is then loaded via the database.

.. code-block:: bash

    info = {"example_key": "example_value"}

    emcee.fit(model=model, analysis=analysis, info=info)

    agg = Aggregator.from_database("path/to/output/database.sqlite")

    info_gen = agg.values("info")

Databases are an extremely powerful feature for users tasked with fitting extremely large datasets as well as fitting
many different models, where the scale of the problem can make the management of the large quantity of results produced
prohibitive. This is especially true on high performance computing facilities, which often have restrictions on the
number of files that a user can store on the machine.

If you'd like to see the ``Aggregator`` in action, checkout the
`database example <https://github.com/Jammy2211/autofit_workspace/blob/master/notebooks/features/database.ipynb>`_ on the
``autofit_workspace``.

The Database Chapter of the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_
provides more details, including how to visualize the results of a model fit fully.