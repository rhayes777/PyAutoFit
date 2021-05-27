.. _multiple_datasets:

Multiple Datasets
-----------------

**NOTE: Graphical models are an in-development feature. This example serves to illustrate what we currently developing , but the API is subject to change. If you are interested in using graphical models I recommend you contact me directly ( https://github.com/Jammy2211 ) so we can discuss how to implement **PyAutoFit** for your use-case.**

For graphical model composition, we saw how to compose a graphical model which fitted a single dataset. In this 
example, we will show how to build a graphical model that fits multiple datasets. 

It is common in statistical inference for us to have a large dataset and not be interested in how well a small aspect 
of the model fits each dataset individually. Instead, we want to fit the complete model to our full dataset and 
determine the global behaviour of the model's fit to every dataset.

Using graphical models, **PyAutoFit** can compose and fit models that have 'local' parameters specific to an individual 
dataset and higher-level model components that fit 'global' parameters. These higher level parameters will have 
conditional dependencies with the local parameters.  

The major selling point of **PyAutoFit**'s graphical modeling framework is the high level of customization it offers,
whereby: 

- Specific ``Analysis`` classes can be defined for fitting differnent local models to different datasets.
- Each pairing of a local model-fit to data can be given its own non-linear search.
- Graphical model networks of any topology can be defined and fitted.

In this example, we demonstrate the API for composing and fitting a graphical model to multiple-datasets, using the 
simple example of fitting noisy 1D Gaussians. In this example, I will explicitly write code that stores each dataset 
as its own Python variable (e.g. data_0, data_1, data_2, etc.), as opposed to a for loop or list. This is to make the 
API shown in this example clear, however examples in the ``autofit_workspace`` will use **PyAutoFit**'s bespoke API 
for setting up a graphical model.

We begin by loading noisy 1D data containing 3 Gaussian's.

.. code-block:: bash

    dataset_path = path.join("dataset", "example_1d")

    dataset_0_path = path.join(dataset_path, "gaussian_x1_0__low_snr")
    data_0 = af.util.numpy_array_from_json(file_path=path.join(dataset_0_path, "data.json"))
    noise_map_0 = af.util.numpy_array_from_json(
        file_path=path.join(dataset_0_path, "noise_map.json")
    )

    dataset_1_path = path.join(dataset_path, "gaussian_x1_1__low_snr")
    data_1 = af.util.numpy_array_from_json(file_path=path.join(dataset_1_path, "data.json"))
    noise_map_1 = af.util.numpy_array_from_json(
        file_path=path.join(dataset_1_path, "noise_map.json")
    )

    dataset_2_path = path.join(dataset_path, "gaussian_x1_2__low_snr")
    data_2 = af.util.numpy_array_from_json(file_path=path.join(dataset_2_path, "data.json"))
    noise_map_2 = af.util.numpy_array_from_json(
        file_path=path.join(dataset_2_path, "noise_map.json")
    )

This is what our three Gaussians look like. They are much lower signal-to-noise than the Gaussian's in other examples. 
We use lower signal-to-noise Gaussian's to demonstrate how fitting graphical models to lower quality data can still 
enable global parameters to be estimated precisely.

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x1_1__low_snr.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x1_2__low_snr.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x1_3__low_snr.png
  :width: 600
  :alt: Alternative text

For each dataset we now create a corresponding ``Analysis`` class. By associating each dataset with an ``Analysis``
class we are therefore associating it with a unique ``log_likelihood_function``. If our dataset had many different
formats (e.g. images) it would be straight forward to write customized ``Analysis`` classes for each dataset.

.. code-block:: bash

    analysis_0 = a.Analysis(data=data_0, noise_map=noise_map_0)
    analysis_1 = a.Analysis(data=data_1, noise_map=noise_map_1)
    analysis_2 = a.Analysis(data=data_2, noise_map=noise_map_2)

We now compose the graphical model we will fit using the ``Model`` and ``Collection`` objects. We begin by setting up a
shared prior for their ``centre`` using a single ``GaussianPrior``. This is passed to a unique ``Model`` for
each ``Gaussian`` and means that all three ``Gaussian``'s are fitted wih the same value of ``centre``. That is, we have
defined our graphical model to have a shared value of ``centre`` when it fits each dataset.

.. code-block:: bash

    centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)

We now set up three ``Model`` objects, each of which contain a ``Gaussian`` that is used to fit each of the
datasets we loaded above. Because all three of these ``Model``'s use the ``centre_shared_prior`` the dimensionality of
parameter space is N=7, corresponding to three ``Gaussians`` with local parameters (``intensity`` and ``sigma``) and
a global parameter value of ``centre``.

.. code-block:: bash

    gaussian_0 = af.Model(m.Gaussian)
    gaussian_0.centre = centre_shared_prior
    gaussian_0.intensity = af.GaussianPrior(mean=10.0, sigma=10.0)
    gaussian_0.sigma = af.GaussianPrior(mean=10.0, sigma=10.0)  # This prior is used by all 3 Gaussians!

    gaussian_1 = af.Model(m.Gaussian)
    gaussian_1.centre = centre_shared_prior
    gaussian_1.intensity = af.GaussianPrior(mean=10.0, sigma=10.0)
    gaussian_1.sigma = af.GaussianPrior(mean=10.0, sigma=10.0)  # This prior is used by all 3 Gaussians!

    gaussian_2 = af.Model(m.Gaussian)
    gaussian_2.centre = centre_shared_prior
    gaussian_2.intensity = af.GaussianPrior(mean=10.0, sigma=10.0)
    gaussian_2.sigma = af.GaussianPrior(mean=10.0, sigma=10.0)  # This prior is used by all 3 Gaussians!

To build our graphical model which fits multiple datasets, we simply pair each model-component to each ``Analysis``
class, so that **PyAutoFit** knows that:

- ``gaussian_0`` fits ``data_0`` via ``analysis_0``.
- ``gaussian_1`` fits ``data_1`` via ``analysis_1``.
- ``gaussian_2`` fits ``data_2`` via ``analysis_2``.

The point where a ``Model`` and ``Analysis`` class meet is called a ``ModelFactor``.

This term is used to denote that we are composing a 'factor graph'. A factor defines a node on this graph where we have
some data, a model, and we fit the two together. The 'links' between these different factors then define the global
model we are fitting **and** the datasets used to fit it.

.. code-block:: bash

    model_factor_0 = g.ModelFactor(prior_model=prior_model_0, analysis=analysis_0)
    model_factor_1 = g.ModelFactor(prior_model=prior_model_1, analysis=analysis_1)
    model_factor_2 = g.ModelFactor(prior_model=prior_model_2, analysis=analysis_2)

We combine our ``ModelFactors`` into one, to compose the factor graph.

.. code-block:: bash

    factor_graph = g.FactorGraphModel(model_factor_0, model_factor_1, model_factor_2)

So, what does our factor graph looks like? Unfortunately, we haven't yet build visualization of this into **PyAutoFit**,
so you'll have to make do with a description for now. The factor graph above is made up of two components:

**Nodes:** these are points on the graph where we have a unique set of data and a model that is made up of a subset of
our overall graphical model. This is effectively the ``ModelFactor`` objects we created above.

**Links:** these define the model components and parameters that are shared across different nodes and thus retain the
same values when fitting different datasets.

.. code-block:: bash

    opt = g.optimise.LaplaceOptimiser(n_iter=3)
    model = factor_graph.optimise(opt)

**Road Map**

The example above which illustrated a simple graphical model where 3 datasets are fitted is fully functional in
**PyAutoFit** and can be ran at the following scripts:

https://github.com/Jammy2211/autofit_workspace/blob/release/scripts/features/graphical_models.py

https://github.com/Jammy2211/autofit_workspace/blob/release/scripts/howtofit/chapter_graphical_models/tutorial_2_graphical_model.py

However, graphical models **are still in beta testing** and I recommend you contact us if you wish to use the
functionality first (https://github.com/Jammy2211).