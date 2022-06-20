.. _result:

Results & Samples
=================

A non-linear search's fit function returns a ``Result`` object:

.. code-block:: python

   analysis = Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(number_of_cores=4)

   result = emcee.fit(model=model, analysis=analysis)

Here, we'll look in detail at what information is contained in the ``Result``.

Info
----

As shown in previous examples, the ``Result`` has an ``info`` attribute which can be printed to get a readable
overview of the results.

.. code-block:: python

    print(result.info)

This gives the following output:

.. code-block:: bash

    Bayesian Evidence                  -58.49183930
    Maximum Log Likelihood             -43.89597618
    Maximum Log Posterior              -43.85625735

    model                              Gaussian (N=3)

    Maximum Log Likelihood Model:

    centre                             49.951
    normalization                      25.287
    sigma                              10.093


    Summary (3.0 sigma limits):

    centre                             49.97 (49.59, 50.38)
    normalization                      25.33 (24.47, 26.25)
    sigma                              10.11 (9.70, 10.48)


    Summary (1.0 sigma limits):

    centre                             49.97 (49.83, 50.10)
    normalization                      25.33 (25.02, 25.63)
    sigma                              10.11 (9.98, 10.24)

Samples
-------

A result contains a ``Samples`` object, which contains information on the non-linear sampling, for example the parameters.
The parameters are stored as a list of lists, where the first entry corresponds to the sample index and second entry
the parameter index.

.. code-block:: python

    samples = result.samples

    print("Final 10 Parameters:")
    print(samples.parameter_lists[-10:])

    print("Sample 10`s third parameter value (Gaussian -> sigma)")
    print(samples.parameter_lists[9][2], "\n")

The Samples class also contains the log likelihood, log prior, log posterior and weight_list of every accepted sample,
where:

- The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise normalized).

- The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log posterior value.

- The log posterior is log_likelihood + log_prior.

- The weight gives information on how samples should be combined to estimate the posterior. The weight values depend on the sampler used, for MCMC samples they are all 1 (e.g. all weighted equally).

Lets inspect the last 10 values of each for the analysis.

.. code-block:: python

    print("Final 10 Log Likelihoods:")
    print(samples.log_likelihood_list[-10:])

    print("Final 10 Log Priors:")
    print(samples.log_prior_list[-10:])

    print("Final 10 Log Posteriors:")
    print(samples.log_posterior_list[-10:])

    print("Final 10 Sample Weights:")
    print(samples.weight_list[-10:], "\n")

Posterior
---------

The ``Result`` object therefore contains the full posterior information of our non-linear search, that can be used for
parameter estimation. The median pdf vector is readily available from the ``Samples`` object, which estimates the every
parameter via 1D marginalization of their PDFs.

.. code-block:: python

    median_pdf_vector = samples.median_pdf_vector

The samples include methods for computing the error estimates of all parameters via 1D marginalization at an input sigma
confidence limit. This can be returned as the size of each parameter error:

.. code-block:: python

    error_vector_at_upper_sigma = samples.error_vector_at_upper_sigma(sigma=3.0)
    error_vector_at_lower_sigma = samples.error_vector_at_lower_sigma(sigma=3.0)

    print("Upper Error values (at 3.0 sigma confidence):")
    print(error_vector_at_upper_sigma)

    print("lower Error values (at 3.0 sigma confidence):")
    print(error_vector_at_lower_sigma, "\n")

They can also be returned at the values of the parameters at their error values:

.. code-block:: python

    vector_at_upper_sigma = samples.vector_at_upper_sigma(sigma=3.0)
    vector_at_lower_sigma = samples.vector_at_lower_sigma(sigma=3.0)

    print("Upper Parameter values w/ error (at 3.0 sigma confidence):")
    print(vector_at_upper_sigma)
    print("lower Parameter values w/ errors (at 3.0 sigma confidence):")
    print(vector_at_lower_sigma, "\n")

**PyAutoFit** includes many visualization tools for plotting the results of a non-linear search, for example we can
make a corner plot of the probability density function (PDF):

.. code-block:: python

    emcee_plotter = aplt.EmceePlotter(samples=result.samples)
    emcee_plotter.corner()

Here is an example of how a PDF estimated for a model appears:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/cornerplot.png
  :width: 600
  :alt: Alternative text

Other Vectors
-------------

The samples contain many useful vectors, including the samples with the highest likelihood and posterior values:

.. code-block:: python

    max_log_likelihood_vector = samples.max_log_likelihood_vector
    max_log_posterior_vector = samples.max_log_posterior_vector

    print("Maximum Log Likelihood Vector:")
    print(max_log_likelihood_vector)

    print("Maximum Log Posterior Vector:")
    print(max_log_posterior_vector, "\n")


Labels
------

These vectors return the results as a list, which means you need to know the parameter ordering. The
list of ``parameter_names`` are available as a property of the ``Samples``, as are ``parameter_labels``
which can be used for labeling figures:

.. code-block:: python

    samples.model.parameter_names
    samples.model.parameter_labels

Instances
---------

``Result``'s can instead be returned as an ``instance``, which is an instance of the model using the Python
classes used to compose it:

.. code-block:: python

    max_log_likelihood_instance = samples.max_log_likelihood_instance

    print("Max Log Likelihood Gaussian Instance:")
    print("Centre = ", max_log_likelihood_instance.centre)
    print("normalization = ", max_log_likelihood_instance.normalization)
    print("Sigma = ", max_log_likelihood_instance.sigma)


For our example problem of fitting a 1D ``Gaussian`` profile, this makes it straight forward to plot
the maximum likelihood model:

.. code-block:: python

    model_data_1d = samples.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
        xvalues=np.arange(data.shape[0])
    )

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data_1d)
    plt.title("Illustrative toy model fit to 1D Gaussian model data.")
    plt.xlabel("x values of 1D profile")
    plt.ylabel("Model data normalization")
    plt.show()
    plt.close()

All methods above are available as an ``instance``:

.. code-block:: python

    median_pdf_instance = samples.median_pdf_instance
    instance_at_upper_sigma = samples.instance_at_upper_sigma
    instance_at_lower_sigma = samples.instance_at_lower_sigma
    error_instance_at_upper_sigma = samples.error_instance_at_upper_sigma
    error_instance_at_lower_sigma = samples.error_instance_at_lower_sigma

An ``instance`` of any accepted sample can be created:

.. code-block:: python

    instance = samples.instance_from_sample_index(sample_index=500)

Bayesian Evidence
-----------------

If a nested sampling non-linear search is used, the Bayesian evidence of the model is also
available which enables model comparison to be performed:

.. code-block:: python

    log_evidence = samples.log_evidence

Result Extensions
-----------------

You might be wondering what else the results contains, as nearly everything we discussed above was a part of its
``samples`` property! The answer is, not much, however the result can be extended to include  model-specific results for
your project.

We detail how to do this in the **HowToFit** lectures, but for the example of fitting a 1D Gaussian we could extend
the result to include the maximum log likelihood profile:

.. code-block:: python

    max_log_likelihood_profile = samples.max_log_likelihood_profile

Database
--------

For large-scaling model-fitting problems to large datasets, the results of the many model-fits performed can be output
and stored in a queryable sqlite3 database. The ``Result`` and ``Samples`` objects have been designed to streamline the
analysis and interpretation of model-fits to large datasets using the database.

The database is described `here <https://pyautofit.readthedocs.io/en/latest/features/database.html>`_

Wrap-Up
-------

More information on the ``Result`` class can be found at the
`results examples <https://github.com/Jammy2211/autofit_workspace/blob/master/notebooks/overview/simple/result.ipynb>`_ on
the ``autofit_workspace``. More details are provided in tutorial 7 or chapter 1 of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_