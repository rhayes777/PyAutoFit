PyAutoFit
=========

**PyAutoFit** is a Python-based probablistic programming language that allows complex model fitting techniques to be
straightforwardly integrated into scientific modeling software. **PyAutoFit** specializes in:

- **Black box** models with complex and expensive log likelihood functions. 
- Fitting **many different model parametrizations** to a data-set. 
- Modeling **extremely large-datasets** with a homogenous fitting procedure. 
- Automating complex model-fitting tasks via **transdimensional model-fitting pipelines**.

API Overview
------------

To illustrate the **PyAutoFit** API, we'll use an illustrative toy model of fitting a one-dimensional Gaussian to
noisy 1D data of a Gaussian's line profile. Here's an example of the data (blue) and the model we'll fit (orange):

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/toy_model_fit.png
  :width: 400
  :alt: Alternative text

We define our model, a 1D Gaussian, by writing a Python class using the format below.

.. code-block:: python

    class Gaussian:

        def __init__(
            self,            # <- PyAutoFit recognises these
            centre = 0.0,    # <- constructor arguments are
            intensity = 0.1, # <- the model parameters of .
            sigma = 0.01,    # <- the Gaussian.
        ):
            self.centre = centre
            self.intensity = intensity
            self.sigma = sigma

    """
    An instance of the Gaussian class will be available during model fitting.
    This method will be used to fit the model to data and compute a likelihood.
    """

    def line_from_xvalues(self, xvalues):

        transformed_xvalues = xvalues - self.centre

        return (self.intensity / (self.sigma * (2.0 * np.pi) ** 0.5)) * \
                np.exp(-0.5 * transformed_xvalues / self.sigma)

**PyAutoFit** recognises that this Gaussian may be treated as a model component whose parameters can be fitted for via
a non-linear search like `emcee <https://github.com/dfm/emcee>`_..

To fit this Gaussian to the data we create an Analysis object, which gives **PyAutoFit** the data and a likelihood
function describing how to fit the data with the model:

.. code-block:: python

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            """
            The 'instance' that comes into this method is an instance of the Gaussian class
            above, with the parameters set to (random) values chosen by the non-linear search.
            """

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("Intensity = ", instance.intensity)
            print("Sigma = ", instance.sigma)

            """
            We fit the data with the Gaussian instance, using its
            "line_from_xvalues" function to create the model data.
            """

            xvalues = np.arange(self.data.shape[0])

            model_data = instance.line_from_xvalues(xvalues=xvalues)
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

We can now fit data to the model using a non-linear search of our choice.

.. code-block:: python

    model = af.PriorModel(Gaussian)

    analysis = a.Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(nwalkers=50, nsteps=2000)

    result = emcee.fit(model=model, analysis=analysis)

The result object contains information on the model-fit, for example the parameter samples, best-fit model and
marginalized probability density functions.

Getting Started
---------------

To get started checkout our `readthedocs <https://pyautofit.readthedocs.io/>`_,
where you'll find our installation guide, a complete overview of **PyAutoFit**'s features, examples scripts and
tutorials and detailed API documentation.

Slack
-----

We're building a **PyAutoFit** community on Slack, so you should contact us on our
`Slack channel <https://pyautofit.slack.com/>`_ before getting started. Here, I give the latest updates on the
software & can discuss how best to use **PyAutoFit** for your science case.

Unfortunately, Slack is invitation-only, so first send me an `email <https://github.com/Jammy2211>`_ requesting an invite.
