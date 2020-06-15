.. _api:

Complex Modeling
----------------

To illustrate advanced *model fitting* with **PyAutoFit**, we extend our problem of fitting a 1D Gaussian line profile
fitting data containing a signal from multiple line profiles, specifically a 1D *Gaussian* and 1D *Exponential* profile.
Example data (blue), including the model-fit we'll perform (orange) and individual Gaussian (red dashed) and
Exponential (green dashed) components are shown below:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/toy_model_fit_x2.png
  :width: 400
  :alt: Alternative text

we again define our 1D Gaussian line profile as a *model component* in **PyAutoFit**:

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

We now define a new *model component*, the 1D Exponential, using the same Python class format:

.. code-block:: bash

    class Exponential:
        def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
            intensity=0.1,  # <- parameters of the Exponential.
            rate=0.01,
        ):

            self.centre = centre
            self.intensity = intensity
            self.rate = rate

        def line_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return self.intensity * np.multiply(
                self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
            )

Before looking at the *Analysis* class, lets look at how we compose the model that we pass to the *non-linear search.
This does not use the *PriorModel* class used in the previous example, but instead uses the *CollectionPriorModel
class:

.. code-block:: bash

    model = af.CollectionPriorModel(gaussian=m.Gaussian, exponential=m.Exponential)

The *CollectionPriorModel* allows us to *compose* models using multiple classes, in the example above using both the
*Gaussian* and *Exponential* classes. The model is defined with 6 free parameters (3 for the *Gaussian*, 3 for the
*Exponential*), thus the dimensionality of non-linear parameter space is 6.

The *model components* given to the *CollectionPriorModel* are also given names, in this case, 'gaussian' and
'exponential'. You can choose whatever name you want and the names are used by the instance passed to the *Analysis*
class:

.. code-block:: bash

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            # The 'instance' that comes into this method is a CollectionPriorModel. It contains instances of every
            # class we instantiated it with, where each instance is named following the names given to the
            # CollectionPriorModel, which in this example is a Gaussian (with name 'gaussian) and Exponential
            # (with name 'exponential'):

            print("Gaussian Instance:")
            print("Centre = ", instance.gaussian.centre)
            print("Intensity = ", instance.gaussian.intensity)
            print("Sigma = ", instance.gaussian.sigma)

            print("Exponential Instance:")
            print("Centre = ", instance.exponential.centre)
            print("Intensity = ", instance.exponential.intensity)
            print("Rate = ", instance.exponential.rate)

            # Get the range of x-values the data is defined on, to evaluate the model of the line profiles.
            xvalues = np.arange(self.data.shape[0])

            # The *instance* variable is a list of our model components. We can iterate over this list, calling their
            # line_from_xvalues and summing the result to compute the summed line profile of our model.

            # The *instance* variable is a list of our model components. We can iterate over this list, calling their
            # line_from_xvalues and summing the result to compute the summed line profile of any model.
            model_data = sum([line.line_from_xvalues(xvalues=xvalues) for line in instance])

            # Fit the model line profile data to the observed data, computing the residuals and chi-squareds.
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

To perform a *model-fit* we again *compose* our model, instantiate the *Analysis* and pass them to the fit method of a
*non-linear search*. In this example, we'll use the nested sampling algorithm *Dynesty*

.. code-block:: bash

    model = af.PriorModel(m.Gaussian)

    analysis = a.Analysis(data=data, noise_map=noise_map)

    dynesty = af.Dynesty()

    result = dynesty.fit(model=model, analysis=analysis)

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

Lets consider exactly what is happening in the *Analysis* class above:

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

We've illustrated how to *compose* models using any number of *model components*. In the next API overview, we'll cover
how to *customize* the model we *compose*, for example specifying priors, fixing parameters and removing regions of
parameter space with assertions.