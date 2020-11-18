.. _model_complex:

Model Composition & Customization
---------------------------------

Lets extend our example of fitting a 1D ``Gaussian`` profile to noisy data, to a problem where the
data contains a signal from two profiles. Specifically, it contains signals from a 1D ``Gaussian`` signal
and 1D symmetric ``Exponential`` signal.

The example ``data`` with errors (black), including the model-fit we'll perform (red) and individual
``Gaussian`` (blue dashed) and ``Exponential`` (orange dashed) components are shown below:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/toy_model_fit_x2.png
  :width: 600
  :alt: Alternative text

we again define our 1D ``Gaussian`` profile as a *model component* in **PyAutoFit**:

.. code-block:: bash

    class Gaussian:
        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            intensity=0.1,  # <- constructor arguments are
            sigma=0.01,     # <- the Gaussian's parameters.
        ):

            self.centre = centre
            self.intensity = intensity
            self.sigma = sigma

        def profile_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return np.multiply(
                np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
                np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
            )

Now lets define a new *model component*, a 1D ``Exponential``, using the same Python class format:

.. code-block:: bash

    class Exponential:
        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            intensity=0.1,  # <- constructor arguments are
            rate=0.01,      # <- the Exponential's parameters.
        ):

            self.centre = centre
            self.intensity = intensity
            self.rate = rate

        def profile_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return self.intensity * np.multiply(
                self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
            )

Before looking at the ``Analysis`` class, lets look at how we *compose* the *model* that we fit the ``data`` with.

Because we now fit multiple *model compoentns*, we do not use the ``PriorModel`` object used in the previous example,
but instead uses the ``CollectionPriorModel`` object:

.. code-block:: bash

    model = af.CollectionPriorModel(gaussian=Gaussian, exponential=Exponential)

The ``CollectionPriorModel`` allows us to *compose* models using multiple classes, in the example above using both the
``Gaussian`` and ``Exponential`` classes. The model is defined with 6 free parameters (3 for the ``Gaussian``, 3 for the
``Exponential``), thus the dimensionality of non-linear parameter space is 6.

The *model components* given to the ``CollectionPriorModel`` are also given names, in this case, 'gaussian' and
'exponential'. You can choose whatever name you want and the names are used by the ``instance`` passed to the ``Analysis``
class:

.. code-block:: bash

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            # The 'instance' that comes into this method is a CollectionPriorModel. It contains
            # instances of every class we instantiated it with, where each instance is named
            # following the names given to the CollectionPriorModel, which in this example is a
            # Gaussian (with name 'gaussian) and Exponential (with name 'exponential'):

            print("Gaussian Instance:")
            print("Centre = ", instance.gaussian.centre)
            print("Intensity = ", instance.gaussian.intensity)
            print("Sigma = ", instance.gaussian.sigma)

            print("Exponential Instance:")
            print("Centre = ", instance.exponential.centre)
            print("Intensity = ", instance.exponential.intensity)
            print("Rate = ", instance.exponential.rate)

            # Get the range of x-values the data is defined on, to evaluate the model of the
            # line profiles.

            xvalues = np.arange(self.data.shape[0])

            # The instance variable is a list of our model components. We can iterate over
            # this list, calling their profile_from_xvalues and summing the result to compute
            # the summed line profile of our model.

            model_data = sum([line.profile_from_xvalues(xvalues=xvalues) for line in instance])

            # Fit the model line profile data to the observed data, computing the residuals and
            # chi-squareds.

            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

Performing the *model-fit* uses the same steps as the previous example, whereby we  *compose* our *model* (now using a
``CollectionPriorModel``), instantiate the ``Analysis`` and pass them a ``NonLinearSearch``. In this example, we'll use
the nested sampling algorithm ``dynesty``, using the ``DynestyStatic`` sampler.

.. code-block:: bash

    model = af.CollectionPriorModel(gaussian=Gaussian, exponential=Exponential)

    analysis = Analysis(data=data, noise_map=noise_map)

    dynesty = af.DynestyStatic(name="example_search")

    result = dynesty.fit(model=model, analysis=analysis)

Now, lets consider how we *customize* the models that we *compose*. To begin, lets *compose* a model using a single
``Gaussian`` with the ``PriorModel`` object:

.. code-block:: bash

    model = af.PriorModel(Gaussian)

By default, the priors on the ``Gaussian``'s parameters are loaded from configuration files. If you have downloaded the
``autofit_workspace`` you can find these files at the path ``autofit_workspace/config/priors``. Alternatively,
you can check them out at this `link <https://github.com/Jammy2211/autofit_workspace/tree/master/config>`_.

Priors can be manually specified as follows:

.. code-block:: bash

    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.intensity = af.LogUniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.sigma = af.GaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf)

These priors will be used by the ``NonLinearSearch`` to determine how it samples parameter space. The ``lower_limit``
and ``upper_limit`` on the ``GaussianPrior`` set the physical limits of values of the parameter, specifying that the
``sigma`` value of the ``Gaussian`` cannot be negative.

We can fit this model, with all new priors, using a ``NonLinearSearch`` as we did before:

.. code-block:: bash

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(name="another_example_search")

    # The model passed here now has updated priors!

    result = emcee.fit(model=model, analysis=analysis)

We can *compose* and *customize* a ``CollectionPriorModel`` as follows:

.. code-block:: bash

    model = af.CollectionPriorModel(gaussian=Gaussian, exponential=Exponential)

    model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.gaussian.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.exponential.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

The model can be *customized* to fix any *parameter* of the model to an input value:

.. code-block:: bash

    model.gaussian.sigma = 0.5

This fixes the ``Gaussian``'s ``sigma`` value to 0.5, reducing the number of free parameters and therefore
dimensionality of *non-linear parameter space* by 1.

We can also link two parameters, such that they always share the same value:

.. code-block:: bash

    model.gaussian.centre = model.exponential.centre

In this model, the ``Gaussian`` and ``Exponential`` will always be centrally aligned. Again, this reduces
the number of free *parameters* by 1.

Finally, assertions can be made on parameters that remove values that do not meet those assertions
from *non-linear parameter space*:

.. code-block:: bash

    model.add_assertion(model.gaussian.sigma > 5.0)
    model.add_assertion(model.gaussian.intensity > model.exponential.intensity)

Here, the ``Gaussian``'s ``sigma`` value must always be greater than 5.0 and its ``intensity`` is greater
than that of the ``Exponential``.

If you'd like to perform the fit shown in this script, checkout the
`complex examples <https://github.com/Jammy2211/autofit_workspace/tree/master/examples/complex>`_ on the
``autofit_workspace``. We provide more details **PyAutoFit** works in the tutorials 5 and 6 of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_