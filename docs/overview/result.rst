.. _api:

Result
------

A *non-linear search*'s fit function returns a *Result* object:

 .. code-block:: bash

    analysis = a.Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(
        number_of_cores=4
    )

    result = emcee.fit(model=model, analysis=analysis)

Here, we'll look in detail at what information is contained in the result. The result contains the model we used to fit
the data:

.. code-block:: bash

    model = result.model

It also contains a *Samples* object, which contains information on the non-linear sampling, for example the parameters:

.. code-block:: bash

    samples = result.samples
    print(samples.parameters)

The parameters are a list of lists of all accepted parameter values sampled by the non-linear search. Also available
are lists of the likelihood, prior, posterior and weight values associated with every sample:

.. code-block:: bash

    print(samples.log_likelihoods)
    print(samples.log_priors)
    print(samples.log_posteriors)
    print(samples.weights)

For MCMC analysis, these can be used perform parameter estimation by binning the samples in a histogram (assuming we
have removed the burn-in phase):

.. code-block:: bash

    samples = result.samples.samples_after_burn_in

    most_probable_vector = [float(np.percentile(samples[:, i], [50])) for i in range(model.prior_count)]

The most probable vector is readily available from the *Samples* object for you convenience (and if a nested sampling
*non-linear search* is used, it will use an appropriate method to estimate the parameters):

.. code-block:: bash

    most_probable_vector = samples.most_probable_vector

The samples contain many useful vectors, including the samples with the highest likelihood and posterior values:

.. code-block:: bash

    max_log_likelihood_vector = samples.max_log_likelihood_vector
    max_log_posterior_vector = samples.max_log_posterior_vector

It also provides methods for computing the error estimates of all parameters at an input sigma confidence limit, which
can be returned at the values of the parameters including their errors or the size of the errors on each parameter:

.. code-block:: bash

    vector_at_upper_sigma = samples.vector_at_upper_sigma(sigma=3.0)
    vector_at_lower_sigma = samples.vector_at_lower_sigma(sigma=3.0)

    error_vector_at_upper_sigma = samples.error_vector_at_upper_sigma(sigma=3.0)
    error_vector_at_lower_sigma = samples.error_vector_at_lower_sigma(sigma=3.0)

Results vectors return the results as a list, which means you need to know the parameter ordering. The list of
parameter names are available as a property of the *Samples*, as are parameter labels which can be used for labeling
figures:

.. code-block:: bash

    samples.parameter_names
    samples.parameter_labels

Results can instead be returned as an instance, which is an instance of the model using the Python classes used to
compose it:

.. code-block:: bash

    max_log_likelihood_instance = samples.max_log_likelihood_instance

    print("Max Log Likelihood Gaussian Instance:")
    print("Centre = ", max_log_likelihood_instance.centre)
    print("Intensity = ", max_log_likelihood_instance.intensity)
    print("Sigma = ", max_log_likelihood_instance.sigma)

For our example problem of fitting a 1D Gaussian line profile, this makes it straight forward to plot the maximum
likelihood model:

.. code-block:: bash

    model_data = samples.max_log_likelihood_instance.line_from_xvalues(
        xvalues=np.arange(data.shape[0])
    )

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data)
    plt.title("Illustrative toy model fit to 1D Gaussian line profile data.")
    plt.xlabel("x values of line profile")
    plt.ylabel("Line profile intensity")
    plt.show()
    plt.close()

All methods above are available as an instance:

.. code-block:: bash

    most_probable_instance = samples.most_probable_instance
    instance_at_upper_sigma = samples.instance_at_upper_sigma
    instance_at_lower_sigma = samples.instance_at_lower_sigma
    error_instance_at_upper_sigma = samples.error_instance_at_upper_sigma
    error_instance_at_lower_sigma = samples.error_instance_at_lower_sigma

An instance of any accepted sample can be created:

.. code-block:: bash

    instance = samples.instance_from_sample_index(sample_index=500)

If a nested sampling *non-linear search* is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed:

.. code-block:: bash

    log_evidence = samples.log_evidence

At this point, you might be wondering what else the results contains - pretty much everything we discussed above was a
part of its *samples* property! For projects which use **PyAutoFit**'s phase API (see here), the *Results* object can
be extended to include model-specific results.

For example, we may extend the results of our 1D Gaussian example to include properties containing the maximum
log likelihood of the summed model data and for every individual line profile in the model:

.. code-block:: bash

    max_log_likelihood_line = results.max_log_likelihood_line
    max_log_likelihood_line_list = results.max_log_likelihood_line_list