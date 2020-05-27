.. _api:

Model Customization
-------------------

In the previous API overviews, we illustrated how to *compose* models using the *PriorModel* and *CollectionPriorModel*
objects. Here, we'll illustrate how we can use these objects to *customize* the model that we fit. We'll use both
examples from the previous examples, that is fitting 1 *Gaussian* line profile and fitting a combined *Gaussian* +
*Exponential* profile.

To begin, lets *compose* a model using a single *Gaussian* and the *PriorModel* object:

.. code-block:: bash

    model = af.PriorModel(m.Gaussian)

By default, the priors on the *Gaussian*'s parameters ae loaded from configuration files. If you have downloaded the
*autofit_workspace* you can find these files at the path *autofit_workspace/config/json_priors*. Alternatively,
you can check them out at this `link <https://github.com/Jammy2211/autofit_workspace/tree/master/config>`_.

Priors can be manually specified as follows:

.. code-block:: bash

    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.intensity = af.LogUniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf)

These priors will be used by the *non-linear search* to determine how it samples parameter space. The lower and upper
limits on the *GaussianPrior* prevent set the physical limits of values of the parameter, in this case specifying that
the *sigma* value of the *Gaussian* cannot be negative.

We can fit this model, with all new priors, using a *non-linear search* as we did before:

.. code-block:: bash

    analysis = a.Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee()

    # The model passed here now has updated priors!

    result = emcee.fit(model=model, analysis=analysis)

We can *compose* and *customize* a *CollectionPriorModel* as follows:

.. code-block:: bash

    model = af.CollectionPriorModel(gaussian=m.Gaussian, exponential=m.Exponential)

    model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

However, as the name suggests, a *CollectionPriorModel* is simply a collection of *PriorModels*. Thus, we can set up a
model with identical priors as follows:

.. code-block:: bash

    gaussian = af.PriorModel(m.Gaussian)

    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

    exponential = af.PriorModel(m.Exponential)

    exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

    model = af.CollectionPriorModel(gaussian=gaussian, exponential=exponential)

Which syntax is better? It really depends on your modeling problem, so feel free to experiment with both!

The model can be *customized* to fix any parameter of the model to a value:

.. code-block:: bash

    model.gaussian.sigma = 0.5

This fixes the second Gaussian's sigma value to 0.5, reducing the number of free parameters and therefore
dimensionality of *non-linear parameter space* by 1.

You can also link two parameters such that they always share the same value:

.. code-block:: bash

    model.gaussian.centre = model.exponential.centre

In this model, the *Gaussian* and *Exponential* will always be centrally aligned. Again, this reduces the number of
free parameters by 1.

Finally, assertions can be made on parameters that remove all values that do not meet those assertions from
*non-linear parameter space*:

.. code-block:: bash

    model.add_assertion(model.gaussian.sigma > 5.0)
    model.add_assertion(model.gaussian.intensity > model.exponential.intensity)

Here, the *Gaussian*'s sigma value must always be greater than 5.0 and its intensity is greater than that of the
*Exponential*.