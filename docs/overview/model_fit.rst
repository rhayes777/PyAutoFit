.. _api:

Model Fit
---------

To illustrate **PyAutoFit** we'll use the example modeling problem of fitting a 1D Gaussian line profile to noisy data
of a 1D Gaussian. Example data (blue), including the model-fit we'll perform (orange), are shown below.

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/toy_model_fit.png
  :width: 400
  :alt: Alternative text

To begin, lets import autofit using the convention below:

.. code-block:: bash

    import autofit as af

we need to define our 1D Gaussian line profile as a *model component* in **PyAutoFit**. A *model component* is a
part of the model used to fit the data and it is associated with set of parameters we fit for during *model-fitting*.

*Model components* are defined using Python classes using the format below, where the class name is the *model
component* name and the constructor arguments are its parameters.

.. code-block:: bash

    class Gaussian:

        def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments
            intensity=0.1,  # <- are the Gaussian's model parameters.
            sigma=0.01,
        ):

            self.centre = centre
            self.intensity = intensity
            self.sigma = sigma

The code above informs defines a **PyAutoFit** *model component* called a *Gaussian*. When it is used in
*model-fitting* it has three free parameters, *centre*, *intensity* and *sigma*.

When we fit the model to data and compute a likelihood instances of the class above will be accessible with specific
values of *centre*, *intensity* and *sigma* chosen. This means that the class's functions will be available to help
compute the likelihood, so lets add a function that generate the 1D line profile from the *Gaussian*.

.. code-block:: bash

    class Gaussian:
        def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments
            intensity=0.1,  # <- are the Gaussian's model parameters.
            sigma=0.01,
        ):

            self.centre = centre
            self.intensity = intensity
            self.sigma = sigma

        def line_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return np.multiply(
                np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
                np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
            )

Now that **PyAutoFit** knows our model, we need to tell it how to fit the model to data. This requires us to extend
the **PyAutoFit** *Analysis* class for our modeling example:

.. code-block:: bash

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            # The 'instance' that comes into this method is an instance of the Gaussian class, which the print
            # statements below illustrate if you run the code!

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("Intensity = ", instance.intensity)
            print("Sigma = ", instance.sigma)

            # Get the range of x-values the data is defined on, to evaluate the model of the Gaussian.
            xvalues = np.arange(self.data.shape[0])

            # Use these xvalues to create model data of our Gaussian.
            model_data = instance.line_from_xvalues(xvalues=xvalues)

            # Fit the model gaussian data to the observed data, computing the residuals and chi-squareds.
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

Lets consider exactly what is happening in the *Analysis* class above.

- The data the model is fitted too is passed into the constructor of the *Analysis* class. Above, only the
  data and noise-map are input, but the constructor can be easily extended to add other data components.

- The log likelihood function receives an *instance* of the model, which in this example is an instance of the
  *Gaussian* class. This *instance* has values for its parameters (*centre*, *intensity* and *sigma*) which are chosen
  by the *non-linear search* used to fit the model, as discussed next.

- The log likelihood function returns a log likelihood value, which the *non-linear* search uses to vary parameter
  values and sample parameter space.

Next, we *compose* our model, set up our *Analysis* and fit the model to the data using a *non-linear search*:

.. code-block:: bash

    model = af.PriorModel(m.Gaussian)

    analysis = a.Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee()

    result = emcee.fit(model=model, analysis=analysis)

Above, we use the class *PriorModel* to compose the model. This is telling **PyAutoFit** that the *Gaussian* class is
to be used as a *model component* where its parameters are to be fitted for by the *non-linear search*. The model is
defined with 3 free parameters, thus the dimensionality of non-linear parameter space is 3.

We perform the fit using the non-linear search algorithm `emcee <https://github.com/dfm/emcee>`_ (for tutorials on what
a non-linear search is and how to use one, checkout ?).

Finally, lets look at the *Result* object returned by our model-fit. It includes lists containing the non-linear
search's parameter samples, the maximum likelihood model, marginalized parameters estimates, errors are so forth:

.. code-block:: bash

    print(result.samples.parameters)
    print(result.samples.max_log_likelihood_vector)
    print(result.samples.median_pdf_vector)
    print(result.samples.error_vector_at_sigma)

It can even return *instances* of the *Gaussian* class using the values of the model results:

.. code-block:: bash

    instance = result.max_log_likelihood_instance

    print("Maximum Likelihood Gaussian Instance:")
    print("Centre = ", instance.centre)
    print("Intensity = ", instance.intensity)
    print("Sigma = ", instance.sigma)

This can be used to straight forwardly plot the model fit to the data:

.. code-block:: bash

    instance = result.max_log_likelihood_instance

    model_data = instance.line_from_xvalues(xvalues=np.arange(data.shape[0]))

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data)

This complete our basic introduction to the **PyAutoFit** API. You may feel ready to adapt the code above to your own
modeling software - if so, go for it! However, I'd advise you continue to the *advanced* API overview first, which
covers how to *compose* and *fit* models using multiple *model components* (e.g. not just one *Gaussian* class).