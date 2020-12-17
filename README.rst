PyAutoFit
=========

**PyAutoFit** is a Python-based probabilistic programming language which:

- Makes it straight forward to compose and fit models using a range of Bayesian inference libraries, such as `emcee <https://github.com/dfm/emcee>`_ and `dynesty <https://github.com/joshspeagle/dynesty>`_.
- Handles the 'heavy lifting' of model fitting, including model composition and customization, outputting results in a structured path format and model-specific visualization.
- Includes bespoke tools for **big-data** analysis, including massively parallel model fitting and database output structures so that large suites of results can be loaded into Jupyter notebooks post-analysis.

Advanced statistical methods in **PyAutoFit** include:

- **Graphical Models**: Combine fits to many different datasets and determine the global parameters of a model, using an `Expectation Propagation Bayesian framework <https://arxiv.org/abs/1412.4869v1>`_.
- **Non-Linear Search Grid-Search**: Simplify the fitting of highly complex parameters spaces by using a grid of non-linear searches, which can be performed simultaneously in an embarrassingly parallel fashion.
- **Transdimensional Pipelines**: Chain together multiple fits of different models to streamline model comparison and automate complex model-fitting tasks.

Getting Started
---------------

To get started checkout our `readthedocs <https://pyautofit.readthedocs.io/>`_,
where you'll find our installation guide, a complete overview of **PyAutoFit**'s features, examples scripts and
tutorials, and the `HowToFit Jupyter notebook tutorials <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_
which introduces new users to **PyAutoFit**.

Installation
------------

**PyAutoFit** requires Python 3.6+ and you can install it via pip or conda (see
`this link <https://pyautofit.readthedocs.io/en/latest/installation/conda.html>`_
for conda instructions).

.. code-block:: bash

    pip install autofit

Next, clone the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_, which includes **PyAutoFit**
configuration files, example scripts and more!

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Finally, run *welcome.py* in the ``autofit_workspace`` to get started!

.. code-block:: bash

   python3 welcome.py

If your installation had an error, please check the
`troubleshooting section <https://pyautofit.readthedocs.io/en/latest/installation/troubleshooting.html>`_ on
our readthedocs.

If you would prefer to Fork / Clone the **PyAutoFit** GitHub repo, please read the
`cloning section <https://pyautofit.readthedocs.io/en/latest/installation/source.html>`_ on our
readthedocs first.

API Overview
------------

To illustrate the **PyAutoFit** API, we'll use an illustrative toy model of fitting a one-dimensional Gaussian to
noisy 1D data. Here's the ``data`` (black) and the model (red) we'll fit:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/toy_model_fit.png
  :width: 400
  :alt: Alternative text

We define our model, a 1D Gaussian by writing a Python class using the format below:

.. code-block:: python

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

        """
        An instance of the Gaussian class will be available during model fitting.

        This method will be used to fit the model to ``data`` and compute a likelihood.
        """

        def profile_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return (self.intensity / (self.sigma * (2.0 * np.pi) ** 0.5)) * \
                    np.exp(-0.5 * transformed_xvalues / self.sigma)

**PyAutoFit** recognises that this Gaussian may be treated as a model component whose parameters can be fitted for via
a `NonLinearSearch` like `emcee <https://github.com/dfm/emcee>`_.

To fit this Gaussian to the ``data`` we create an Analysis object, which gives **PyAutoFit** the ``data`` and a
``log_likelihood_function`` describing how to fit the ``data`` with the model:

.. code-block:: python

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            """
            The 'instance' that comes into this method is an instance of the Gaussian class
            above, with the parameters set to values chosen by the non-linear search.
            """

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("Intensity = ", instance.intensity)
            print("Sigma = ", instance.sigma)

            """
            We fit the ``data`` with the Gaussian instance, using its
            "profile_from_xvalues" function to create the model data.
            """

            xvalues = np.arange(self.data.shape[0])

            model_data = instance.profile_from_xvalues(xvalues=xvalues)
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

We can now fit our model to the ``data`` using a ``NonLinearSearch``:

.. code-block:: python

    model = af.PriorModel(Gaussian)

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(nwalkers=50, nsteps=2000)

    result = emcee.fit(model=model, analysis=analysis)

The ``result`` contains information on the model-fit, for example the parameter samples, maximum log likelihood
model and marginalized probability density functions.

Support
-------

Support for installation issues and integrating your modeling software with **PyAutoFit** is available by
`raising an issue on the autofit_workspace GitHub page <https://github.com/Jammy2211/autofit_workspace/issues>`_. or
joining the **PyAutoFit** `Slack channel <https://pyautofit.slack.com/>`_, where we also provide the latest updates on
**PyAutoFit**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.