PyAutoFit
=========

**PyAutoFit** is a Python-based probablistic programming language that allows Bayesian inference techniques to be
straightforwardly integrated into scientific modeling software. **PyAutoFit** specializes in:

- **Black box** models with complex and expensive log likelihood functions. 
- Fitting **many different model parametrizations** to a data-set. 
- Modeling **extremely large-datasets** with a homogenous fitting procedure. 
- Automating complex model-fitting tasks via **transdimensional model-fitting pipelines**.

API Overview
============

**PyAutoFit** interfaces with Python classes and non-linear sampling packages such as
[PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html). Lets take a one-dimensional
Gaussian as our moodel:

.. code-block:: python

    class Gaussian:

        def __init__(
            self,
            centre = 0.0,    # <- PyAutoFit recognises these constructor arguments are the model
            intensity = 0.1, # <- parameters of the Gaussian.
            sigma = 0.01,
        ):
            self.centre = centre
            self.intensity = intensity
            self.sigma = sigma

    # An instance of the Gaussian class will be available to PyAutoFit, meaning methods which help us fit the model to
    # data are accessible.

    def line_from_xvalues(self, xvalues):

        transformed_xvalues = np.subtract(xvalues, self.centre)

        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )

**PyAutoFit** recognises that this Gaussian may be treated as a model component whose parameters could be fitted for
by a non-linear search.

To fit this Gaussian to some data we create an Analysis object, which gives **PyAutoFit** the data and tells it how to
fit it with the model:

.. code-block:: python

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            # The 'instance' that comes into this method is an instance of the Gaussian class above, with the parameters
            # set to values chosen by the non-linear search.

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("Intensity = ", instance.intensity)
            print("Sigma = ", instance.sigma)

            # Below, we fit the data with the Gaussian instance, using its "line_from_xvalues" function to create the
            # model data.

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

    multi_nest = af.MultiNest(
        n_live_points=50,
        sampling_efficiency=0.5
        )

    result = multi_nest.fit(model=model, analysis=analysis)

The result object contains informationon the model-fit, for example the parameter samples, best-fit model and
marginalized probability density functions.

Features
========

Model Customization
-------------------

It is straight forward to parameterize, customize and fit models made from multiple components. Below, we extend the
example above to include a second Gaussian and an Expoenntial profile, with user-specified priors and a centre aligned
with the first Gaussian:

.. code-block:: python

    # The model can be setup with multiple classes and before passing it to a phase and
    # we can customize the model parameters.

    model = af.CollectionPriorModel(
        gaussian_0=Gaussian, gaussian_1=Gaussian, exponential=Exponential
    )

    # This aligns the centres of the Gaussian and Exponential, reducing the number of free parameters by 1.
    model.gaussian_0.centre = model.exponential.centre

    # This fixes the Gaussian's sigma value to 0.5, reducing the number of free parameters by 1.
    model.gaussian_1.sigma = 0.5

    # We can customize the priors on any model parameter.
    model.gaussian_0.intensity = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    model.exponential.rate = af.GaussianPrior(mean=0.1, sigma=0.05)

    # We can make assertions on parameters which remove regions of parameter space where these are not valid
    model.add_assertion(model.exponential.intensity > 0.5)

    analysis = a.Analysis(data=data, noise_map=noise_map)

    multi_nest = af.MultiNest()

    result = multi_nest.fit(model=model, analysis=analysis)

Aggregation
-----------

For fits to large data-sets **PyAutoFit** provides tools to manipulate the vast library of results output. 

Lets pretend we performed the Gaussian fit above to 100 indepedent data-sets. Every **PyAutoFit** output contains
metadata meaning that we can immediately load it via the **aggregator** into a Python script or Jupyter notebook:

.. code-block:: python

    # Lets pretend we've used a Phase object to fit 100 different datasets with the same model. The results of
    # these 100 fits are in a structured output format in this folder.
    output_path = "/path/to/gaussian_x100_fits/"

    # To create an instance of the aggregator, we pass it the output path above. The aggregator will detect that
    # 100 fits using a specific phase have been performed and that their results are in this folder.
    agg = af.Aggregator(directory=str(output_path))

    # The aggregator can now load results from these fits. The command below loads results as instances of the
    # Samples class, which provides the non-linear seach parameter samples, log-likelihood values, etc.
    samples = agg.values("samples")

    # The results of all 100 non-linear searches are now available. The command below creates a list of
    # instances of the best-fit model parameters of all 100 model fits (many other results are available, e.g.
    # marginalized 1D parameter estimates, errors, Bayesian evidences, etc.).
    instances = [output.max_log_likelihood_instance for samps in samples]

    # These are instances of the 'model-components' defined using the PyAutoFit Python class format illustrated
    # in figure 1. For the Gaussian class, each instance in this list is an instance of this class and its parameters
    # are accessible.
    print("Instance Parameters \n")
    print("centre = ", instances[0].centre)
    print("intensity = ", instances[0].intensity)
    print("sigma = ", instances[0].sigma)

    # The aggregator can be customized to interface with model-specific aspects of a project like the data and
    # fitting procedure. Below, the aggregator has been set up to provide instances of all 100 datasets.
    datasets = agg.values("dataset")

    # If the datasets are fitted with many different phases (e.g. with different models), the aggregator's filter
    # tool can be used to load results of a specific phase (and therefore model).
    phase_name = "phase_example"
    samples = agg.filter(agg.phase == phase_name).values("samples")

Phases
------

For long-term software development projects, users can write a **PyAutoFit** *phase* module, which contain all
information about the model-fitting process, e.g. the data, model and analysis. This allows **PyAutoFit** to provide
the software project with a clean and intuitive interface for model-fitting whilst taking care of the 'heavy lifting'
that comes with performming model fitting, including:

- Outputting results in a structured path format.
- Providing on-the-fly model output and visusalization.
- Augmenting and customizing the dataset used to fit the model.
- Building and fitting complex models composed of many model components.

Below is an example of how straight forward the Gaussian model fit above is to perform once a phase module has been
written for it:

.. code-block:: python

    # To perform the fit we set up a phase, which takes as input the model and non-linear search. It sets up the
    # Analysis class 'behind the scenes', and uses its name to structure the results output to hard disk.

    phase = af.Phase(model=af.Gaussian, phase_name="phase_example", non_linear_class=af.MultiNest)

    # To perform a model fit, we simply pass the phase's 'run' method a dataset to the phase..

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
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear optimizer
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
                gaussian_0=phase1.result.instance.gaussians.gaussian_0, # <- Use the Gaussian fitted in phase 1
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
                gaussian_0=phase1.result.model.gaussians.gaussian_0, # <- use phase 1 Gaussian results.
                gaussian_1=phase2.result.model.gaussians.gaussian_1, # <- use phase 2 Gaussian results.
            ),
            non_linear_class=af.MultiNest,
        )

        return toy.Pipeline(pipeline_name, phase1, phase2, phase3)

[PyAutoLens](https://github.com/Jammy2211/PyAutoLens) shows a real-use case of transdimensional modeling, fitting
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
[Slack channel](https://pyautofit.slack.com/) before getting started. Here, I will give you the latest updates on the
software & discuss how best to use **PyAutoFit** for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.


Documentation & Installation
----------------------------

The PyAutoLens documentation can be found at our `readthedocs  <https://pyautofit.readthedocs.io/en/master>`_,
including instructions on `installation <https://pyautofit.readthedocs.io/en/master/installation.html>`_.

Support & Discussion
====================

If you're having difficulty with installation, model fitting, or just want a chat, feel free to message us on our
[Slack channel](https://pyautofit.slack.com/).

Contributing
============

If you have any suggestions or would like to contribute please get in touch.

Credits
=======

**Developers:**

[Richard Hayes](https://github.com/rhayes777) - Lead developer

[James Nightingale](https://github.com/Jammy2211) - Lead developer
