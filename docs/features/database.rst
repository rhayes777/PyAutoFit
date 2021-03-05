.. _database:

Database
--------

The default behaviour of **PyAutoFit** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check the model-fitting results and visualization. For small model-fitting
tasks this is sufficient, however many users have a need to perform many model fits to very large datasets, making
the manual inspection of results too time consuming.

PyAutoFit's database feature outputs all model-fitting results as a sqlite3 (https://docs.python.org/3/library/sqlite3.html)
relational database, such that all results can be efficiently loaded into a Jupyter notebook or Python script for
inspection, analysis and interpretation. This database supports advanced querying, so that specific
model-fits (e.g., which fit a certain model or dataset) can be loaded.

Lets suppose that we have performed 100 model-fits to 100 1D Gaussians, and when we ran **PyAutoFit** we told it
to not output the results in folders on our hard-disc but instead to a ``.sqlite`` file. We can then load these results
into a Python script or Jupyter notebook using the ``Aggregator`` as follows:

.. code-block:: bash

    agg = Aggregator.from_database("database_gaussian_x100_fits.sqlite")

We can now use the ``Aggregator`` to inspect the results of all model-fits. For example, we can load the ``Samples``
object of all 100 model-fits, which contains information on the best-fit model, posterior, Bayesian evidence, etc.

.. code-block:: bash

    samples = agg.values("samples")

This object (and all objects load by the ``Aggregator``) are returned as a generator, (as opposed to a list,
dictionary or other Python type). This is because generators do not store large arrays or classes in memory until they
are used, ensuring that when we are manipulating large sets of results we do not run out of memory!

Below, we use the samples generator to create a list of the maximum log likelihood of every model-fit and print it:

.. code-block:: bash

    for samps in agg.values("samples"):

        print(max(samps.log_likelihood))

We can iterate over the samples to print the maximum log likelihood model of every fit: results above are returned as
lists containing entries for every model-fit, in this case 100 fits:

.. code-block:: bash

    for samps in agg.values("samples"):

        instance = samps.max_log_likelihood_instance

        print("Maximum Likelihood Model-fit \n")
        print("centre = ", instance.centre)
        print("intensity = ", instance.intensity)
        print("sigma = ", instance.sigma)


The ``Aggregator`` can be used in Python scripts, however we recommend users adopt Jupyter Notebooks when
using the ``aggregator``. Notebooks allow results to be inspected and visualized with immediate feedback,
such that one can more readily interpret the results.

The ``Aggregator`` contains tools for querying the database and filtering the results, for example to load subsets of
model-fits. The simplest way to do this is to require that the database path the results contains a certain string
(or strings) which are passed to the model-fit before it is performed.

For example, provided we included this string when performing the model fit, we could filter for the string ``gaussian_10``,
meaning we would only load the results of the *model-fit* to the 10th ``Gaussian`` in our dataset:

.. code-block:: bash

    agg_filter = agg.filter(
        agg.directory.contains("gaussian_10")
    )

Lets pretend you fitted a dataset independently 3 times, and wish to combine these fits into one ``Samples``
object such that methods that return parameter estimates or errors use the combined fit. This can be done
by simply adding the ``Samples`` objects together:

.. code-block:: bash

    samples = list(agg.values("samples"))

    samples = samples[0] + samples[1] + samples[2]

    samples.median_pdf_instance

If you'd like to see the ``Aggregator`` in action, checkout the
`database example <https://github.com/Jammy2211/autofit_workspace/blob/master/notebooks/features/database.ipynb>`_ on the
``autofit_workspace``.

The Database Chapter of the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_
provides more details, including how to reperform visualization and advanced database querying.