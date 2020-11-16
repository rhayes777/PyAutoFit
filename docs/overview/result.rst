.. _result:

Results & Samples
-----------------

A ``NonLinearSearch``'s fit function returns a ``Result`` object:

.. code-block:: bash

   analysis = Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(number_of_cores=4)

   result = emcee.fit(model=model, analysis=analysis)

Here, we'll look in detail at what information is contained in the ``Result``. The result contains the model
we used to fit the data:

.. code-block:: bash

    model = result.model

It also contains a ``Samples`` object, which contains information on the non-linear sampling, for example
the ``parameters``:

.. code-block:: bash

    samples = result.samples
    print(samples.parameters)

The ``parameters`` are a list of lists of all accepted parameter values sampled by the ``NonLinearSearch``. Also
available are lists of the ``log_likelihoods``, ``log_priors``, ``log_posteriors`` and ``weights`` associated
with every sample:

.. code-block:: bash

    print(samples.log_likelihoods)
    print(samples.log_priors)
    print(samples.log_posteriors)
    print(samples.weights)

For MCMC analysis, these are used to perform parameter estimation by binning the samples in a histogram
(assuming we have removed the burn-in phase):

.. code-block:: bash

    samples = result.samples.samples_after_burn_in

    median_pdf_vector = [float(np.percentile(samples[:, i], [50])) for i in range(model.prior_count)]

The ``median_pdf_vector`` is readily available from the ``Samples`` object for you convenience (and
if a nested sampling ``NonLinearSearch`` is used, it will use an appropriate method to estimate the
parameters):

.. code-block:: bash

    median_pdf_vector = samples.median_pdf_vector

The ``Samples`` contain many useful vectors, including the ``Samples`` with the highest likelihood and
posterior values:

.. code-block:: bash

    max_log_likelihood_vector = samples.max_log_likelihood_vector
    max_log_posterior_vector = samples.max_log_posterior_vector

It also provides methods for computing the error estimates of all parameters at an input ``sigma``
confidence limit, which can be returned at the values of the parameters including their errors
or the size of the errors on each parameter:

.. code-block:: bash

    vector_at_upper_sigma = samples.vector_at_upper_sigma(sigma=3.0)
    vector_at_lower_sigma = samples.vector_at_lower_sigma(sigma=3.0)

    error_vector_at_upper_sigma = samples.error_vector_at_upper_sigma(sigma=3.0)
    error_vector_at_lower_sigma = samples.error_vector_at_lower_sigma(sigma=3.0)

These vectors return the results as a list, which means you need to know the parameter ordering. The
list of ``parameter_names`` are available as a property of the ``Samples``, as are ``parameter_labels``
which can be used for labeling figures:

.. code-block:: bash

    samples.model.parameter_names
    samples.model.parameter_labels

``Result``'s can instead be returned as an ``instance``, which is an instance of the model using the Python
classes used to compose it:

.. code-block:: bash

    max_log_likelihood_instance = samples.max_log_likelihood_instance

    print("Max Log Likelihood Gaussian Instance:")
    print("Centre = ", max_log_likelihood_instance.centre)
    print("Intensity = ", max_log_likelihood_instance.intensity)
    print("Sigma = ", max_log_likelihood_instance.sigma)

For our example problem of fitting a 1D ``Gaussian`` profile, this makes it straight forward to plot
the maximum likelihood model:

.. code-block:: bash

    model_data = samples.max_log_likelihood_instance.profile_from_xvalues(
        xvalues=np.arange(data.shape[0])
    )

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data)
    plt.title("Illustrative toy model fit to 1D Gaussian line profile data.")
    plt.xlabel("x values of line profile")
    plt.ylabel("Line profile intensity")
    plt.show()
    plt.close()

All methods above are available as an ``instance``:

.. code-block:: bash

    median_pdf_instance = samples.median_pdf_instance
    instance_at_upper_sigma = samples.instance_at_upper_sigma
    instance_at_lower_sigma = samples.instance_at_lower_sigma
    error_instance_at_upper_sigma = samples.error_instance_at_upper_sigma
    error_instance_at_lower_sigma = samples.error_instance_at_lower_sigma

An ``instance`` of any accepted sample can be created:

.. code-block:: bash

    instance = samples.instance_from_sample_index(sample_index=500)

If a nested sampling ``NonLinearSearch`` is used, the Bayesian evidence of the model is also
available which enables model comparison to be performed:

.. code-block:: bash

    log_evidence = samples.log_evidence

At this point, you might be wondering what else the ``Result``'s contains, pretty much everything we
discussed above was a part of its ``Samples`` property! For projects which use **PyAutoFit**'s phase
API (see `here <https://pyautofit.readthedocs.io/en/latest/overview/phase.html>`_), the ``Result``'s object
can be extended to include model-specific results.

For example, we may extend the results of our 1D ``Gaussian`` example to include properties like the
``max_log_likelihood_profile`` (e.g. the 1D model data of the best-fit profile) or a list of these
profiles for every individual line profile in the model:

.. code-block:: bash

    max_log_likelihood_profile = results.max_log_likelihood_profile
    max_log_likelihood_profile_list = results.max_log_likelihood_profile_list

More information on the ``Result`` class can be found at the
`results examples <https://github.com/Jammy2211/autofit_workspace/blob/master/examples/simple/result.py>`_ on
the ``autofit_workspace``. More details are provided in chapter 2 of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_