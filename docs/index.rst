Probabilistic Programming
=========================

Probabilistic programming languages (PPLs) have enabled contemporary statistical inference techniques to be applied
to a diverse range of problems across academia and industry. Packages such as
`PyMC3 <https://github.com/pymc-devs/pymc3>`_, `Pyro <https://github.com/pyro-ppl/pyro>`_ and
`STAN <https://github.com/stan-dev/stan>`_ offer general-purpose frameworks where users can specify a generative
model and fit it to data using a variety of non-linear fitting techniques. Each package is specialized to problems
of a certain nature, for example generalized linear models. In these packages the *model* is composed of linear
equations which are easily expressed syntactically, such that the interface of each PPL offers an expressive way to
define the *model* and extensions can be implemented in an intuitive and straightforward way.

Why PyAutoFit?
==============

**PyAutoFit** is a PPL whose core design is providing a direct interface with the model, data, fitting procedure and 
results, providing a more complete management of the *model-fitting* than other PPLs. **PyAutoFit** is particularly
suited too:

- **Big Data Projects**: All **PyAutoFit** model-fitting results are written as a database so they can
  easily be loaded in a Jupyter notebook after model-fitting is complete for analysis and inspection.
- **High Performance Computing Projects**: **PyAutoFit** has deidicated functionality for scaling up model fitting
  to HPC architectures.
- **Long term software development projects**: **PyAutoFit** includes many tools for managing model composition,
  fitting and outputting results that streamline development.

**PyAutoFit** also support many advanced statistic methods, such as *transdimensional modeling*,
*advanced model comparison* and *advanced grid-searches*. Checkout the 'advanced features' tab of the readthedocs
for more information

How does PyAutoFit Work?
========================

You can try **PyAutoFit** now by going to the `overview Jupyter Notebook on our
Binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/664a86aa84ddf8fdf044e2e4e7db21876ac1de91?filepath=overview.ipynb>`_.
This allows you to run the code that is described below.

Model components are written as Python classes, allowing **PyAutoFit** to define the *model* and
associated *parameters* in an expressive way that is tied to the modeling software's API. Here is a simple example of
how a *model* representing a 1D Gaussian is written:

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

A *model* fit then only requires that a **PyAutoFit** ``Analysis`` class is written, which combines the data, model and
likelihood function and defines how the *model-fit* is performed using a `NonLinearSearch`
(e.g. `dynesty <https://github.com/joshspeagle/dynesty>`_, `emcee <https://github.com/dfm/emcee>`_
or `PySwarms <https://pyswarms.readthedocs.io/en/latest/>`_).

Lets take a look at an example ``Analysis`` class:

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

The ``Analysis`` class provides a model specific interface between **PyAutoFit** and the modeling software, allowing
it to handle the 'heavy lifting' that comes with writing *model-fitting* software. This includes interfacing with the
non-linear search, model-specific visualization during and outputting results in a database.

Performing a fit with a non-linear search, for example ``emcee``, is performed as follows:

.. code-block:: python

    model = af.PriorModel(Gaussian)

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(name="example_search", nwalkers=50, nsteps=2000)

    result = emcee.fit(model=model, analysis=analysis)


Model Abstraction and Composition
=================================

For many modeling problems the model comprises abstract *model components* representing objects or processes in a
physical system. For example, our child project `PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_,  where
*model components* represent the light and mass of galaxies. For these problems the likelihood function is typically a
sequence of numerical processes (e.g. convolutions, Fourier transforms, linear algebra) and extensions to the *model* 
often requires the addition of new *model components* in a way that is non-trivially included in the fitting process
and likelihood function. Existing PPLs have tools for these problems (e.g. `black-box' likelihood functions in PyMC3),
however these solutions decouple *model composition* from the data and fitting procedure, making the *model* 
less expressive, restricting *model customization* and reducing flexibility in how the *model-fit* is performed.

By writing *model components* as ``Python`` classes, the *model* and its associated *parameters* are defined in an
expressive way that is tied to the modeling software’s API. *Model composition* with **PyAutoFit** allows complex
*models* to be built from these individual components, abstracting the details of how they change *model-fitting*
procedure from the user. *Models* can be fully customized, allowing adjustment of individual parameter priors, the
fixing or coupling of parameters between *model components* and removing regions of parameter space via parameter
assertions. Adding new *model components* to a **PyAutoFit** project is straightforward, whereby adding a new
``Python`` class means it works within the entire modeling framework. **PyAutoFit** is therefore ideal for
problems where there is a desire to *compose*, *fit* and *compare* many similar (but slightly different) models to a
single dataset, with the **Aggregator** including tools to facilitate this.

To see this in action, checkout the `overview section <https://pyautofit.readthedocs.io/en/latest/overview/model_fit.html>`_
of our readthedocs and the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_
on how to integrate **PyAutoFit** into your modeling software. More statistically minded readers may be interested
in **PyAutoFit**'s advanced statistical methods, such
as `transdimensional pipielines <https://pyautofit.readthedocs.io/en/latest/advanced/pipelines.html>`_.

.. toctree::
   :caption: Overview:
   :maxdepth: 1
   :hidden:

   overview/model_fit
   overview/model_complex
   overview/non_linear_search
   overview/result
   overview/aggregator

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

   general/installation
   general/workspace
   general/configs
   general/software
   general/citations
   general/future
   general/credits

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   howtofit/howtofit
   howtofit/chapter_1_introduction
   howtofit/chapter_phase_api

.. toctree::
   :caption: API Reference:
   :maxdepth: 1
   :hidden:

   api/api

.. toctree::
   :caption: Advanced:
   :maxdepth: 1
   :hidden:

   advanced/phase
   advanced/pipelines