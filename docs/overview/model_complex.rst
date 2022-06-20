.. _model_complex:

Model Composition
=================

Lets extend our example of fitting a 1D ``Gaussian`` profile to a problem where the data contains a signal from
two 1D profiles. Specifically, it contains signals from a 1D ``Gaussian`` signal and 1D symmetric ``Exponential``.

Data
----

The example ``data`` with errors (black), including the model-fit we'll perform (red) and individual
``Gaussian`` (blue dashed) and ``Exponential`` (orange dashed) components are shown below:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/toy_model_fit_x2.png
  :width: 600
  :alt: Alternative text

Model Composition
-----------------

we again define our 1D ``Gaussian`` profile as a *model component* in **PyAutoFit**:

.. code-block:: python

    class Gaussian:
        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            normalization=0.1,  # <- constructor arguments are
            sigma=0.01,     # <- the Gaussian's parameters.
        ):

            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma

        def model_data_1d_via_xvalues_from(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return np.multiply(
                np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
                np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
            )

Now lets define a new *model component*, a 1D ``Exponential``, using the same Python class format:

.. code-block:: python

    class Exponential:
        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            normalization=0.1,  # <- constructor arguments are
            rate=0.01,      # <- the Exponential's parameters.
        ):

            self.centre = centre
            self.normalization = normalization
            self.rate = rate

        def model_data_1d_via_xvalues_from(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return self.normalization * np.multiply(
                self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
            )

We are now fitting multiple *model components*, therefore we create each component using the ``Model`` object we
used in the previous tutorial and put them together in a ``Collection`` to build the overall model.

.. code-block:: python

    gaussian = af.Model(Gaussian)
    exponential = af.Model(Exponential)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

The ``Collection`` allows us to *compose* models using multiple classes. This model is defined with 6 free
parameters (3 for the ``Gaussian``, 3 for the ``Exponential``), thus the dimensionality of non-linear parameter space
is 6.

We can confirm this by printing ``info`` attribute of the model:

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Model Prior Count = 6
    gaussian
        centre                         UniformPrior, lower_limit = 0.0, upper_limit = 100.0
        normalization                  LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                          UniformPrior, lower_limit = 0.0, upper_limit = 25.0
    exponential
        centre                         UniformPrior, lower_limit = 0.0, upper_limit = 100.0
        normalization                  LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
        rate                           UniformPrior, lower_limit = 0.0, upper_limit = 1.0

Analysis
--------

The *model components* given to the ``Collection`` were also given names, in this case, ``gaussian`` and
``exponential``.

You are free to choose whichever names you want;  the names are used to pass the ``instance`` to the ``Analysis`` class:

.. code-block:: python

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            """
            The 'instance' that comes into this method is a Collection. It contains
            instances of every class we instantiated it with, where each instance is named
            following the names given to the Collection, which in this example is a
            Gaussian (with name 'gaussian) and Exponential (with name 'exponential'):
            """

            print("Gaussian Instance:")
            print("Centre = ", instance.gaussian.centre)
            print("normalization = ", instance.gaussian.normalization)
            print("Sigma = ", instance.gaussian.sigma)

            print("Exponential Instance:")
            print("Centre = ", instance.exponential.centre)
            print("normalization = ", instance.exponential.normalization)
            print("Rate = ", instance.exponential.rate)

            """
            Get the range of x-values the data is defined on, to evaluate the model of the
            1D profiles.
            """

            xvalues = np.arange(self.data.shape[0])

            """
            The instance variable is a list of our model components. We can iterate over
            this list, calling their model_data_1d_via_xvalues_from and summing the result to compute
            the summed 1D model data of the 1D profiles in our model.
            """

            model_data_1d = sum([
                profile_1d.model_data_1d_via_xvalues_from(xvalues=xvalues) for profile_1d in instance
            ])

            """
            Fit the 1D model data to the observed data, computing the residuals and
            chi-squared.
            """

            residual_map = self.data - model_data_1d
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

Model Fit
---------

Performing the *model-fit* uses the same steps as the previous example, whereby we  *compose* our *model* (now using a
``Collection``), instantiate the ``Analysis`` and pass them a non-linear search.

In this example, we'll use the nested sampling algorithm ``dynesty``, using the ``DynestyStatic`` sampler.

.. code-block:: python

    model = af.Collection(gaussian=Gaussian, exponential=Exponential)

    analysis = Analysis(data=data, noise_map=noise_map)

    search = af.DynestyStatic(name="example_search")

    result = search.fit(model=model, analysis=analysis)

The ``Result`` object's ``info`` attribute confirms that a model with 6 parameters was fitted successfully:

.. code-block:: python

    print(result.info)

This gives the following output:

.. code-block:: bash

    Bayesian Evidence                  -89.59876054
    Maximum Log Likelihood             -38.90532783
    Maximum Log Posterior              -38.90532783
    
    model                              CollectionPriorModel (N=6)
        gaussian                       Gaussian (N=3)
        exponential                    Exponential (N=3)
    
    Maximum Log Likelihood Model:
    
    gaussian
        centre                         49.682
        normalization                  27.690
        sigma                          10.174
    exponential
        centre                         50.360
        normalization                  38.395
        rate                           0.049
    
    
    Summary (3.0 sigma limits):
    
    gaussian
        centre                         49.70 (48.85, 50.70)
        normalization                  27.55 (22.07, 33.23)
        sigma                          10.16 (9.65, 10.63)
    exponential
        centre                         50.30 (49.55, 50.84)
        normalization                  38.51 (35.59, 41.13)
        rate                           0.05 (0.04, 0.05)
    
    
    Summary (1.0 sigma limits):
    
    gaussian
        centre                         49.70 (49.45, 49.93)
        normalization                  27.55 (25.76, 29.44)
        sigma                          10.16 (9.98, 10.35)
    exponential
        centre                         50.30 (50.11, 50.47)
        normalization                  38.51 (37.65, 39.35)
        rate                           0.05 (0.05, 0.05)

Model Priors
------------

Now, lets consider how we *customize* the models that we *compose*. To begin, lets *compose* a model using a single
``Gaussian`` with the ``Model`` object:

.. code-block:: python

    gaussian = af.Model(Gaussian)

By default, the priors on the ``Gaussian``'s parameters (which are printed above) are loaded from configuration files.

If you have downloaded the ``autofit_workspace`` you can find these files at the path ``autofit_workspace/config/priors``.
Alternatively, you can check them out at this `link <https://github.com/Jammy2211/autofit_workspace/tree/master/config>`_.

Priors can be manually specified as follows:

.. code-block:: python

    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.LogUniformPrior(lower_limit=0.0, upper_limit=1e2)
    gaussian.sigma = af.GaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf)

These priors will be used by the non-linear search to determine how it samples parameter space. The ``lower_limit``
and ``upper_limit`` on the ``GaussianPrior`` set the physical limits of values of the parameter, specifying that the
``sigma`` value of the ``Gaussian`` cannot be negative.

Printing the `model.info` attribute shows the priors have been updated:

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    centre                             UniformPrior, lower_limit = 0.0, upper_limit = 100.0
    normalization                      LogUniformPrior, lower_limit = 0.01, upper_limit = 100.0
    sigma                              GaussianPrior, mean = 10.0, sigma = 5.0

We can fit this model, with all new priors, using a non-linear search as we did before:

.. code-block:: python

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(name="another_example_search")

    # The model passed here now has updated priors!

    result = emcee.fit(model=gaussian, analysis=analysis)

We can *compose* and *customize* the priors of multiple model components as follows:

.. code-block:: python

    gaussian = af.Model(Gaussian)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)

    exponential = af.Model(Exponential)
    exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

Model Customization
-------------------

The model can be *customized* to fix any *parameter* of the model to an input value:

.. code-block:: python

    gaussian.sigma = 0.5

This fixes the ``Gaussian``'s ``sigma`` value to 0.5, reducing the number of free parameters and therefore
dimensionality of *non-linear parameter space* by 1.

We can also link two parameters, such that they always share the same value:

.. code-block:: python

    model.gaussian.centre = model.exponential.centre

In this model, the ``Gaussian`` and ``Exponential`` will always be centrally aligned. Again, this reduces
the number of free *parameters* by 1.

Finally, assertions can be made on parameters that remove values that do not meet those assertions
from *non-linear parameter space*:

.. code-block:: python

    gaussian.add_assertion(gaussian.sigma > 5.0)
    gaussian.add_assertion(gaussian.normalization > exponential.normalization)

Here, the ``Gaussian``'s ``sigma`` value must always be greater than 5.0 and its ``normalization`` is greater
than that of the ``Exponential``.

Cookbooks
---------

The model cookbook section provides a concise API reference to all of the model composition tools above, as well
as illustrating other features and alternative ways to compose a model.

 - `cookbook 1: Basics  <https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_1_basics.html>`_

 - `cookbook 2: Collections  <https://pyautofit.readthedocs.io/en/latest/cookbooks/cookbook_2_collections.html>`_

Advanced Model Composition And Cookbooks
----------------------------------------

Advanced model component in **PyAutoFit** includes:

- Models which fit multiple datasets where specific parameters vary across the datasets (see `cookbook 4 <https://pyautofit.readthedocs.io/en/latest/model_cookbooks/variable_across_data.html>`_).

- Multi-level models which compose models via hierarchies of Python classes (see `cookbook 3 <https://pyautofit.readthedocs.io/en/latest/model_cookbooks/multi_level.html>`_).

- Models which are composed from the previous of previous model fits, to build automated model-fitting pipelines (see `cookbook 6 <https://pyautofit.readthedocs.io/en/latest/model_cookbooks/model_linking.html>`_).

Wrap Up
-------

If you'd like to perform the fit shown in this script, checkout the
`complex examples <https://github.com/Jammy2211/autofit_workspace/tree/master/notebooks/overview/complex>`_ on the
``autofit_workspace``. We provide more details **PyAutoFit** works in the tutorials 5 and 6 of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_