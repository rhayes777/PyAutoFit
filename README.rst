PyAutoFit
=========

**PyAutoFit** is a Python-based probablistic programming language that allows complex model fitting techniques to be
straightforwardly integrated into scientific modeling software. **PyAutoFit** specializes in:

- **Black box** models with complex and expensive log likelihood functions. 
- Fitting **many different model parametrizations** to a data-set. 
- Modeling **extremely large-datasets** with a homogenous fitting procedure. 
- Automating complex model-fitting tasks via **transdimensional model-fitting pipelines**.

API Overview
============

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

    # An instance of the Gaussian class will be available during model fitting.
    # This method will be used to fit the model to data and compute a likelihood.

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

            # The 'instance' that comes into this method is an instance of the Gaussian class
            # above, with the parameters set to (random) values chosen by the non-linear search.

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("Intensity = ", instance.intensity)
            print("Sigma = ", instance.sigma)

            # We fit the data with the Gaussian instance, using its
            # "line_from_xvalues" function to create the model data.

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

Features
========

Model Customization
-------------------

It is straight forward to parameterize, customize and fit models made from multiple components. Below, we extend the
example above to include a second Gaussian, with user-specified priors and a centre aligned with the first Gaussian:

.. code-block:: python

    # Using a CollectionPriorModel object the model can be composed of multiple model classes.
    model = af.CollectionPriorModel(
        gaussian_0=m.Gaussian, gaussian_1=m.Gaussian,
    )

    # This aligns the centres of the two Gaussian model components, reducing the number
    # of free parameters by 1.
    model.gaussian_0.centre = model.gaussian_1.centre

    # This fixes the second Gaussian's sigma value to 0.5, reducing the number of
    # free parameters by 1.
    model.gaussian_1.sigma = 0.5

    # We can customize the priors on any model parameter.
    model.gaussian_0.intensity = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    model.gaussian_0.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)

    # We can make assertions on parameters which remove regions of parameter space.
    model.add_assertion(model.gaussian_1.sigma > 5.0)

Aggregation
-----------

For fits to large data-sets **PyAutoFit** provides tools to manipulate the vast library of results output. 

Lets pretend we performed the Gaussian fit above to 100 different data-sets. Every **PyAutoFit** output contains
metadata allowing us to load it via the **aggregator** into a Python script or Jupyter notebook:

.. code-block:: python

    # Lets pretend we fitted 100 different datasets with the same model
    # and the results of these 100 fits are in the output folder:
    output_path = "/path/to/gaussian_x100_fits/"

    # We create an instance of the aggregator by passing it the output path above.
    # The aggregator detects that 100 unique fits have been performed.
    agg = af.Aggregator(directory=str(output_path))

    # To load the result of every fit we call the aggregator's values method. This
    # creates 100 instances of the Samples class, providing parameter samples,
    # log-likelihood, weights, etc or every fit.
    samples = agg.values("samples")

    # This list of Samples provides detailed information on every fit. Lets create
    # 100 instances of the Gaussian class using each fit's maximum log-likelihood
    # model. (many results are available, e.g. marginalized 1D parameter estimates,
    # errors, Bayesian evidences, etc.).
    instances = [samps.max_log_likelihood_instance for samps in agg.values("samples")]

    # These are instance of the 'model-components' defined using the Python class
    # format illustrated above.
    print("First Gaussian Instance Parameters \n")
    print("centre = ", instances[0].centre)
    print("intensity = ", instances[0].intensity)
    print("sigma = ", instances[0].sigma)

    # The aggregator interfaces with many aspects of a model fit. Below, the aggregator
    # loads instances of all 100 datasets.
    datasets = agg.values("dataset")

    # If fits using many different models were performed, the aggregator's filter
    # tool can be used to load results of a specific model.
    dataset_name = "gaussian_dataset_0"
    samples = agg.filter(agg.dataset == dataset_name).values("samples")

Phases
------

For long-term software development projects, users can write a **PyAutoFit** *phase* module, which contain all
information about the model-fitting process, e.g. the data, model and analysis. This allows **PyAutoFit** to provide
the software with a clean and intuitive interface for model-fitting whilst taking care of the 'heavy lifting' that
comes with performming model fitting, including:

- Outputting results in a structured path format.
- Providing on-the-fly model output and visualization.
- Augmenting and customizing the dataset used to fit the model.
- Building and fitting complex models composed of many model components.
- Advanced aggregator tools for filtering and analysing model-fit results.

Below is an example of how the *Phase* API allows the Gaussian model fit to be performed using just 2 lines of Python:

.. code-block:: python

    # Set up a phase, which takes a name, the model and a non-linear search.
    # The phase contains Analysis class 'behind the scenes', as well as taking
    # care of results output, visualization, etc.

    phase = af.Phase(phase_name="phase_example", model=Gaussian, non_linear_class=af.Emcee)

    # To perform a model fit, we simply run the phase with a dataset.

    result = phase.run(dataset=dataset)

HowToFit
---------

Included with **PyAutoFit** is the **HowToFit** lecture series, which provides an introduction to non-linear searches
and model-fitting with **PyAutoFit**. It can be found in the workspace & consists of 1 chapter:

- **Introduction** - How to perform non-linear model-fitting with **PyAutoFit** and write a *phase* module to exploit
                     **PyAutoFits**'s advanced modeling features.

Workspace
---------

**PyAutoFit** comes with a workspace, which can be found `here <https://github.com/Jammy2211/autofit_workspace>`_ &
which includes:

- **API** - Illustrative scripts of the **PyAutoFit** interface to help set up and perform a model-fit.
- **Config** - Configuration files which customize **PyAutoFits**'s behaviour.
- **HowToFit** - The **HowToFit** lecture series.
- **Output** - Where the **PyAutoFit** analysis and visualization are output.


Transdimensional Modeling
=========================

In transdimensional modeling many different models are paramertized and fitted to the same data-set.  

This is performed using **transdimensional model-fitting pipelines**, which break the model-fit into a series of
**linked non-linear searches**, or phases. Initial phases fit simplified realizations of the model, whose results are
used to initialize fits using more complex models in later phases.

Fits of complex models with large dimensionality can therefore be broken down into a series of
**bite-sized model fits**, allowing even the most complex model fitting problem to be **fully automated**.

Lets illustrate this with an example fitting two 2D Gaussians:

![alt text](https://github.com/rhayes777/PyAutoFit/blob/master/gaussian_example.png)

We're going to fit each with the 2D Gaussian profile above. Traditional approaches would fit both Gaussians
simultaneously, making parameter space more complex, slower to sample and increasing the risk that we fail to locate
the global maxima solution. With **PyAutoFit** we can instead build a transdimensional model fitting pipeline which
breaks the the analysis down into 3 phases:

1) Fit only the left Gaussian.
2) Fit only the right Gaussian, using the model of the left Gaussian from phase 1 to reduce blending.
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear search
   searches parameter space.

.. code-block:: python

    def make_pipeline():

        # In phase 1, we will fit the Gaussian on the left.

        phase1 = af.Phase(
            phase_name="phase_1__left_gaussian",
            gaussians=af.CollectionPriorModel(gaussian_0=Gaussian),
            non_linear_class=af.MultiNest,
        )

        # In phase 2, we will fit the Gaussian on the right, where the best-fit Gaussian
        # resulting from phase 1 above fits the left-hand Gaussian.

        phase2 = af.Phase(
            phase_name="phase_2__right_gaussian",
            phase_folders=phase_folders,
            gaussians=af.CollectionPriorModel(
                # Use the Gaussian fitted in Phase 1:
                gaussian_0=phase1.result.instance.gaussians.gaussian_0,
                gaussian_1=Gaussian,
            ),
            non_linear_class=af.MultiNest,
        )

        # In phase 3, we fit both Gaussians, using the results of phases 1 and 2 to
        # initialize their model parameters.

        phase3 = af.Phase(
            phase_name="phase_3__both_gaussian",
            phase_folders=phase_folders,
            gaussians=af.CollectionPriorModel(
                # use phase 1 Gaussian results as priors.
                gaussian_0=phase1.result.model.gaussians.gaussian_0,
                # use phase 2 Gaussian results as priors.
                gaussian_1=phase2.result.model.gaussians.gaussian_1,
            ),
            non_linear_class=af.MultiNest,
        )

        return toy.Pipeline(pipeline_name, phase1, phase2, phase3)

`PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_ shows a real-use case of transdimensional modeling, fitting
galaxy-scale strong gravitational lenses. In this example pipeline, a 5-phase **PyAutoFit** pipeline breaks-down the
fit of 5 diferent models composed of over 10 unique model components and 10-30 free parameters.

Future
======

The following features are planned for 2020 - 2021:

- **Bayesian Model Comparison** - Determine the most probable model via the Bayesian log evidence.
- **Generalized Linear Models** - Fit for global trends to model fits to large data-sets.
- **Hierarchical modeling** - Combine fits over multiple data-sets to perform hierarchical inference.
- **Time series modelling** - Fit temporally varying models using fits which marginalize over time.
- **Approximate Bayesian Computation** - Likelihood-free modeling.
- **Transdimensional Sampling** - Sample non-linear parameter spaces with variable numbers of model components and parameters.

Slack
=====

We're building a **PyAutoFit** community on Slack, so you should contact us on our
`Slack channel <https://pyautofit.slack.com/>`_ before getting started. Here, I will give you the latest updates on the
software & discuss how best to use **PyAutoFit** for your science case.

Unfortunately, Slack is invitation-only, so first send me an `email <https://github.com/Jammy2211>`_ requesting an invite.


Documentation & Installation
----------------------------

The PyAutoLens documentation can be found at our `readthedocs  <https://pyautofit.readthedocs.io/en/master>`_,
including instructions on `installation <https://pyautofit.readthedocs.io/en/master/installation.html>`_.

Support & Discussion
====================

If you're having difficulty with installation, model fitting, or just want a chat, feel free to message us on our
`Slack channel <https://pyautofit.slack.com/>`_.

Contributing
============

If you have any suggestions or would like to contribute please get in touch.

Credits
=======

**Developers:**

`Richard Hayes <https://github.com/rhayes777>`_ - Lead developer

`James Nightingale <https://github.com/Jammy2211>`_ - Lead developer
