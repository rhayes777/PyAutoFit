.. _result:

Results & Samples
=================

A non-linear search's fit function returns a ``Result`` object:

.. code-block:: python

   analysis = Analysis(data=data, noise_map=noise_map)

   search = af.Emcee()

   result = search.fit(model=model, analysis=analysis)

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

    print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
    print(samples.log_likelihood_list[9])
    print(samples.log_prior_list[9])
    print(samples.log_posterior_list[9])
    print(samples.weight_list[9])

Instances
---------

The `Samples` contains many results which are returned as an instance of the model, using the Pythono class structure
used above to compose it.

For example, we can return the model parameters corresponding to the maximum log likelihood fit:

.. code-block:: python

    max_lh_instance = samples.max_log_likelihood()

    print("Max Log Likelihood Gaussian Instance:")
    print("Centre = ", max_lh_instance.centre)
    print("normalization = ", max_lh_instance.normalization)
    print("Sigma = ", max_lh_instance.sigma)

For complex models, with a large number of model components and parameters, this offers a readable API to interpret
the results.

Let us consider the the complex model (composed of a ``Gaussian`` and ``Exponential``) illustrated in the previous
tutorial:

.. code-block:: python

    gaussian = af.Model(Gaussian)
    exponential = af.Model(Exponential)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

Here is how the result is returned:

.. code-block:: python

    max_lh_instance = samples.max_log_likelihood()

    print("Max Log Likelihood `Gaussian` Instance:")
    print("Centre = ", max_lh_instance.gaussian.centre)
    print("Normalization = ", max_lh_instance.gaussian.normalization)
    print("Sigma = ", max_lh_instance.gaussian.sigma, "\n")

    print("Max Log Likelihood Exponential Instance:")
    print("Centre = ", max_lh_instance.exponential.centre)
    print("Normalization = ", max_lh_instance.exponential.normalization)
    print("Sigma = ", max_lh_instance.exponential.rate, "\n")

For our example problem of fitting a 1D ``Gaussian`` profile, this makes it straight forward to plot
the maximum likelihood model:

.. code-block:: python

    model_data = max_lh_instance.model_data_1d_via_xvalues_from(
        xvalues=np.arange(data.shape[0])
    )

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data)
    plt.title("Illustrative model fit to 1D `Gaussian` data.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()

Vectors
-------

All results can alternatively be returned as a 1D vector of values, by passing `as_instance=False`:

.. code-block:: python

    max_lh_vector = samples.max_log_likelihood(as_instance=False)
    print("Max Log Likelihood Model Parameters: \n")
    print(max_lh_vector, "\n\n")

Labels
------

These vectors return the results as a list, which means you need to know the parameter ordering. The
list of ``parameter_names`` are available as a property of the ``Samples``, as are ``parameter_labels``
which can be used for labeling figures:

.. code-block:: python

    model = samples.model

    print(model.parameter_names)
    print(model.parameter_labels)

Posterior
---------

The ``Result`` object therefore contains the full posterior information of our non-linear search, that can be used for
parameter estimation.

The median pdf vector is readily available from the ``Samples`` object, which estimates the every parameter via
1D marginalization of their PDFs.

.. code-block:: python

    median_pdf_instance = samples.median_pdf()

    print("Median PDF `Gaussian` Instance:")
    print("Centre = ", median_pdf_instance.centre)
    print("Normalization = ", median_pdf_instance.normalization)
    print("Sigma = ", median_pdf_instance.sigma, "\n")

Errors
------

The samples include methods for computing the error estimates of all parameters via 1D marginalization at an input sigma
confidence limit. This can be returned as the size of each parameter error:

.. code-block:: python

    errors_at_upper_sigma_instance = samples.errors_at_upper_sigma(sigma=3.0)
    errors_at_lower_sigma_instance = samples.errors_at_lower_sigma(sigma=3.0)

    print("Upper Error values (at 3.0 sigma confidence):")
    print("Centre = ", errors_at_upper_sigma_instance.centre)
    print("Normalization = ", errors_at_upper_sigma_instance.normalization)
    print("Sigma = ", errors_at_upper_sigma_instance.sigma, "\n")

    print("lower Error values (at 3.0 sigma confidence):")
    print("Centre = ", errors_at_lower_sigma_instance.centre)
    print("Normalization = ", errors_at_lower_sigma_instance.normalization)
    print("Sigma = ", errors_at_lower_sigma_instance.sigma, "\n")

They can also be returned at the values of the parameters at their error values:

.. code-block:: python

    values_at_upper_sigma_instance = samples.values_at_upper_sigma(sigma=3.0)
    values_at_lower_sigma_instance = samples.values_at_lower_sigma(sigma=3.0)

    print("Upper Parameter values w/ error (at 3.0 sigma confidence):")
    print("Centre = ", values_at_upper_sigma_instance.centre)
    print("Normalization = ", values_at_upper_sigma_instance.normalization)
    print("Sigma = ", values_at_upper_sigma_instance.sigma, "\n")

    print("lower Parameter values w/ errors (at 3.0 sigma confidence):")
    print("Centre = ", values_at_lower_sigma_instance.centre)
    print("Normalization = ", values_at_lower_sigma_instance.normalization)
    print("Sigma = ", values_at_lower_sigma_instance.sigma, "\n")

Search Plots
------------

**PyAutoFit** includes many visualization tools for plotting the results of a non-linear search, for example we can
make a corner plot of the probability density function (PDF):

.. code-block:: python

    emcee_plotter = aplt.EmceePlotter(samples=result.samples)
    emcee_plotter.corner()

Here is an example of how a PDF estimated for a model appears:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/main/docs/images/cornerplot.png
  :width: 600
  :alt: Alternative text

Other Results
--------------

The samples contain many useful vectors, including the samples with the highest posterior values:

.. code-block:: python

    max_log_posterior_instance = samples.max_log_posterior()

    print("Maximum Log Posterior Vector:")
    print("Centre = ", max_log_posterior_instance.centre)
    print("Normalization = ", max_log_posterior_instance.normalization)
    print("Sigma = ", max_log_posterior_instance.sigma, "\n")


All methods above are available as a vector:

.. code-block:: python

    median_pdf_instance = samples.median_pdf(as_instance=False)
    values_at_upper_sigma = samples.values_at_upper_sigma(sigma=3.0, as_instance=False)
    values_at_lower_sigma = samples.values_at_lower_sigma(sigma=3.0, as_instance=False)
    errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0, as_instance=False)
    errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0, as_instance=False)

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any model -- below we create an instance of the last accepted model.

.. code-block:: python

    instance = samples.from_sample_index(sample_index=-1)

    print("Gaussian Instance of last sample")
    print("Centre = ", instance.centre)
    print("Normalization = ", instance.normalization)
    print("Sigma = ", instance.sigma, "\n")

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
`results examples <https://github.com/Jammy2211/autofit_workspace/blob/main/notebooks/overview/simple/result.ipynb>`_ on
the ``autofit_workspace``. More details are provided in tutorial 7 or chapter 1 of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_