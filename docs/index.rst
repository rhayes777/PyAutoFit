Probabilistic Programming
=========================

Probabilistic programming languages provide a framework that allows users to easily specify a probabilistic
model and perform inference automatically. PyAutoFit is a Python based probabilistic programming language for the
fully Bayesian analysis of extremely large datasets which:

- Makes it simple to compose and fit multi-level models using a range of Bayesian inference libraries, such as `emcee <https://github.com/dfm/emcee>`_ and `dynesty <https://github.com/joshspeagle/dynesty>`_.

- Handles the 'heavy lifting' that comes with model-fitting, including model composition & customization, outputting results, visualization and parameter inference.

- Is built for *big-data* analysis, whereby results are output as a sqlite database which can be queried after model-fitting is complete.

**PyAutoFit** supports advanced statistical methods such as `massively parallel non-linear search grid-searches <https://pyautofit.readthedocs.io/en/latest/features/search_grid_search.html>`_, `chaining together model-fits <https://pyautofit.readthedocs.io/en/latest/features/search_chaining.html>`_  and `sensitivity mapping <https://pyautofit.readthedocs.io/en/latest/features/sensitivity_mapping.html>`_.

Try it now
----------

You can try **PyAutoFit** now by going to the `introduction Jupyter Notebook on our
Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=introduction.ipynb>`_, which runs
**PyAutoFit** in a web browser without installation.

Why PyAutoFit?
--------------

**PyAutoFit** is developed by Astronomers for fitting large imaging datasets of galaxies. We found that existing
probabilistic programming languages (e.g `PyMC3 <https://github.com/pymc-devs/pymc3>`_, `Pyro <https://github.com/pyro-ppl/pyro>`_,
`STAN <https://github.com/stan-dev/stan>`_) were not suited to the type of model fitting problems Astronomers faced,
for example:

- Fitting large and homogenous datasets with an identical model fitting procedure, with tools for processing the large libraries of results output.

- Problems where likelihood evaluations are expensive, leading to run times of days per fit and necessitating support for massively parallel computing.

- Fitting many different models to the same dataset with tools that streamline model comparison.

How does PyAutoFit Work?
========================

To illustrate the **PyAutoFit** API, we'll use an illustrative toy model of fitting a one-dimensional Gaussian to
noisy 1D data. Here's the ``data`` (black) and the model (red) we'll fit:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/files/toy_model_fit.png
  :width: 400

Model components are written as Python classes, allowing **PyAutoFit** to define the *model* and
associated *parameters* in an expressive way that is tied to the modeling software's API.

Here is a simple example of how a *model* representing a 1D Gaussian is written:

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

        """
        An instance of the Gaussian class will be available during model fitting.

        This method will be used to fit the model to ``data`` and compute a likelihood.
        """

        def model_data_1d_via_xvalues_from(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return (self.normalization / (self.sigma * (2.0 * np.pi) ** 0.5)) * \
                    np.exp(-0.5 * transformed_xvalues / self.sigma)

A model-fit requires that a **PyAutoFit** ``Analysis`` class is written, which combines the data and model via
likelihood function:

.. code-block:: python

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            """
            The 'instance' that comes into this method is an instance of the Gaussian class
            above, with the *parameters* set to values chosen by the non-linear search.
            """

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("normalization = ", instance.normalization)
            print("Sigma = ", instance.sigma)

            """
            We fit the ``data`` with the Gaussian instance, using its
            "model_data_1d_via_xvalues_from" function to create the model data.
            """

            xvalues = np.arange(self.data.shape[0])

            model_data_1d = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)
            residual_map = self.data - model_data_1d
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

The ``Analysis`` class provides a model specific interface between **PyAutoFit** and the modeling software, allowing
it to handle the 'heavy lifting' that comes with writing *model-fitting* software. This includes interfacing with the
non-linear search, model-specific visualization during and outputting results to a queryable sqlite database.

Performing a fit with a non-linear search, for example ``emcee``, is performed as follows:

.. code-block:: python

    model = af.Model(Gaussian)

    analysis = Analysis(data=data, noise_map=noise_map)

    search = af.Emcee(name="example_search", nwalkers=50, nsteps=2000)

    result = search.fit(model=model, analysis=analysis)

The ``result`` contains information on the model-fit, for example the parameter samples, maximum log likelihood
model and marginalized probability density functions.

Model Abstraction and Composition
=================================

For many model fitting problems the model comprises abstract *model components* representing objects or processes in a
physical system. For example, our child project `PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_,  where
*model components* represent the light and mass of galaxies. For these problems the likelihood function is typically a
sequence of numerical processes (e.g. convolutions, Fourier transforms, linear algebra) and extensions to the *model* 
often requires the addition of new *model components* in a way that is non-trivially included in the fitting process
and likelihood function. Existing PPLs have tools for these problems, however they decouple *model composition* from the
data and fitting procedure, making the *model* less expressive, restricting *model customization* and reducing
flexibility in how the *model-fit* is performed.

By writing *model components* as ``Python`` classes, the *model* and its associated *parameters* are defined in an
expressive way that is tied to the modeling software’s API. *Model composition* with **PyAutoFit** allows complex
*models* to be built from these individual components, abstracting the details of how they change *model-fitting*
procedure from the user. *Models* can be fully customized, allowing adjustment of individual parameter priors, the
fixing or coupling of parameters between *model components* and removing regions of parameter space via parameter
assertions. Adding new *model components* to a **PyAutoFit** project is straightforward, whereby adding a new
``Python`` class means it works within the entire modeling framework. **PyAutoFit** is therefore ideal for
problems where there is a desire to *compose*, *fit* and *compare* many similar (but slightly different) models to a
single dataset, with **Database** tools available to facilitate this.

The `overview section <https://pyautofit.readthedocs.io/en/latest/overview/model_fit.html>`_ gives a run-down of
**PyAutoFit**'s core features and the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_
provides new users with a more detailed introduction to **PyAutoFit**.

.. toctree::
   :caption: Overview:
   :maxdepth: 1
   :hidden:

   overview/the_basics

.. toctree::
   :caption: Cookbooks:
   :maxdepth: 1
   :hidden:

   cookbooks/model
   cookbooks/analysis
   cookbooks/search
   cookbooks/result
   cookbooks/configs
   cookbooks/multiple_datasets
   cookbooks/database
   cookbooks/multi_level_model

.. toctree::
   :caption: Features:
   :maxdepth: 1
   :hidden:

   features/graphical
   features/search_chaining
   features/search_grid_search
   features/sensitivity_mapping

.. toctree::
   :caption: Installation:
   :maxdepth: 1
   :hidden:

   installation/overview
   installation/conda
   installation/pip
   installation/source
   installation/troubleshooting

.. toctree::
   :caption: General:
   :maxdepth: 1
   :hidden:

   general/workspace
   general/configs
   general/roadmap
   general/software
   general/citations
   general/credits

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   howtofit/howtofit
   howtofit/chapter_1_introduction
   howtofit/chapter_database
   howtofit/chapter_graphical_models

.. toctree::
   :caption: Science Examples:
   :maxdepth: 1
   :hidden:

   science_examples/astronomy

.. toctree::
   :caption: API Reference:
   :maxdepth: 1
   :hidden:

   api/model
   api/priors
   api/analysis
   api/searches
   api/plot
   api/samples
   api/database
   api/source

