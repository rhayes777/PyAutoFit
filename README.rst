PyAutoFit: Classy Probabilistic Programming
===========================================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.02550/status.svg
   :target: https://doi.org/10.21105/joss.02550

|binder| |JOSS|

`Installation Guide <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautofit.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/master?filepath=introduction.ipynb>`_ |
`HowToFit <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_

**PyAutoFit** is a Python-based probabilistic programming language which:

- Makes it simple to compose and fit mult-level models using a range of Bayesian inference libraries, such as `emcee <https://github.com/dfm/emcee>`_ and `dynesty <https://github.com/joshspeagle/dynesty>`_.

- Handles the 'heavy lifting' that comes with model-fitting, including model composition & customization, outputting results, model-specific visualization and posterior analysis.

- Is built for *big-data* analysis, whereby results are output as a sqlite database which can be queried after model-fitting is complete.

**PyAutoFit** supports advanced statistical methods such as `massively parallel non-linear search grid-searches <https://pyautofit.readthedocs.io/en/latest/features/search_grid_search.html>`_, `chaining together model-fits <https://pyautofit.readthedocs.io/en/latest/features/search_chaining.html>`_  and `sensitivity mapping <https://pyautofit.readthedocs.io/en/latest/features/sensitivity_mapping.html>`_.

Getting Started
---------------

The following links are useful for new starters:

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/master?filepath=introduction.ipynb>`_, where you can try **PyAutoFit** in a web browser (without installation).

- `The PyAutoFit readthedocs <https://pyautofit.readthedocs.io/en/latest>`_, which includes an `installation guide <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_ and an overview of **PyAutoFit**'s core features.

- `The autofit_workspace GitHub repository <https://github.com/Jammy2211/autofit_workspace>`_, which includes example scripts and the `HowToFit Jupyter notebook tutorials <https://github.com/Jammy2211/autofit_workspace/tree/master/notebooks/howtofit>`_ which give new users a step-by-step introduction to **PyAutoFit**.

Why PyAutoFit?
--------------

**PyAutoFit** began as an Astronomy project for fitting large imaging datasets of galaxies after the developers found that existing PPLs
(e.g., `PyMC3 <https://github.com/pymc-devs/pymc3>`_, `Pyro <https://github.com/pyro-ppl/pyro>`_, `STAN <https://github.com/stan-dev/stan>`_)
were not suited to the model fitting problems many Astronomers faced. This includes:

- Efficiently analysing large and homogenous datasets with an identical model fitting procedure, with tools for processing the large libraries of results output.

- Problems where likelihood evaluations are expensive (e.g. run times of days per model-fit), necessitating highly customizable model-fitting pipelines with support for massively parallel computing.

- Fitting many different models to the same dataset with tools that streamline model comparison.

If these challenges sound familiar, then **PyAutoFit** may be the right software for your model-fitting needs!

API Overview
------------

To illustrate the **PyAutoFit** API, we'll use an illustrative toy model of fitting a one-dimensional Gaussian to
noisy 1D data. Here's the ``data`` (black) and the model (red) we'll fit:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/files/toy_model_fit.png
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

        This method will be used to fit the model to data and compute a likelihood.
        """

        def profile_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return (self.intensity / (self.sigma * (2.0 * np.pi) ** 0.5)) * \
                    np.exp(-0.5 * (transformed_xvalues / self.sigma) ** 2.0)

**PyAutoFit** recognises that this Gaussian may be treated as a model component whose parameters can be fitted for via
a non-linear search like `emcee <https://github.com/dfm/emcee>`_.

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

We can now fit our model to the ``data`` using a non-linear search:

.. code-block:: python

    model = af.Model(Gaussian)

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(nwalkers=50, nsteps=2000)

    result = emcee.fit(model=model, analysis=analysis)

The ``result`` contains information on the model-fit, for example the parameter samples, maximum log likelihood
model and marginalized probability density functions.

Support
-------

Support for installation issues, help with Fit modeling and using **PyAutoFit** is available by
`raising an issue on the GitHub issues page <https://github.com/rhayes777/PyAutoFit/issues>`_.

We also offer support on the **PyAutoFit** `Slack channel <https://pyautoFit.slack.com/>`_, where we also provide the 
latest updates on **PyAutoFit**. Slack is invitation-only, so if you'd like to join send 
an `email <https://github.com/Jammy2211>`_ requesting an invite.