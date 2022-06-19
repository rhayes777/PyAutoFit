.. _cookbook_5_model_linking

Model Linking
=============

Prerequisites
-------------

You should be familiar with the search chaining API detailed in the following scripts and docs:

Overview
--------

Search chaining allows one to perform back-to-back non-linear searches to fit a dataset, where the model complexity
increases after each fit.

To perform search chaining, **PyAutoFit** has tools for passing the results of one model-fit from one fit to the next,
and change its parameterization between each fit.

This cookbook is a concise reference to the model linking API.

Model-Fit
---------

.. code-block:: python

    We perform a quick model-fit, to create a ``Result`` object which has the attributes necessary to illustrate the model
    linking API.

    model = af.Collection(gaussian=af.ex.Gaussian, exponential=af.ex.Exponential)

    dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    dynesty = af.DynestyStatic(
        name="cookbook_6_model_linking",
    )

    result = dynesty.fit(model=model, analysis=analysis)

Instance & Model
----------------

The result object has two key attributes for model linking:

 - ``instance``: The maximum log likelihood instance of the model-fit, where every parameter is therefore a float.

 - ``model``: An attribute which represents how the result can be passed as a model-component to the next fit (the
 details of how its priors are passed are given in full below).

Below, we create a new model using both of these attributes, where:

 - All of the ``gaussian`` model components parameters are passed via the ``instance`` attribute and therefore fixed to
 the inferred maximum log likelihood values (and are not free parameters in the model).

  - All of the ``exponential`` model components parameters are passed via the ``model`` attribute and therefore are free
  parameters in the model.

The new model therefore has 3 free parameters and 3 fixed parameters.

.. code-block:: python

    model = af.Collection(gaussian=af.ex.Gaussian, exponential=af.ex.Exponential)

    model.gaussian = result.instance.gaussian
    model.exponential = result.model.exponential

The ``model.info`` attribute shows that the parameter and prior passing has occurred as described above.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    gaussian
        centre                         49.62558008533755
        normalization                  27.75178980320016
        sigma                          10.13156736768096
    exponential
        centre                         GaussianPrior, mean = 50.31368408748868, sigma = 20.0
        normalization                  GaussianPrior, mean = 38.46597213618446, sigma = 19.23298606809223
        rate                           GaussianPrior, mean = 0.04924782286498935, sigma = 0.024623911432494674


We can print the priors of the exponenital:

.. code-block:: python

    print("Exponential Model Priors \n")
    print("centre = ", model.exponential.centre)
    print("normalization = ", model.exponential.normalization)
    print("rate = ", model.exponential.rate)

This gives the following output:

.. code-block:: bash

    centre =  GaussianPrior, mean = 50.31368408748868, sigma = 20.0
    normalization =  GaussianPrior, mean = 38.46597213618446, sigma = 19.23298606809223
    rate =  GaussianPrior, mean = 0.04924782286498935, sigma = 0.024623911432494674

How are the priors set via model linking? The full description is quite long, therefore it is attatched to the
bottom of this script so that we can focus on the model linking API.

Component Specification
-----------------------

Model linking can be performed on any component of a model, for example to only pass specific parameters as
an ``instance`` or ``model``.

.. code-block:: python

    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = result.instance.gaussian.centre
    gaussian.normalization = result.model.gaussian.normalization
    gaussian.sigma = result.instance.gaussian.sigma

    exponential = af.Model(af.ex.Exponential)

    exponential.centre = result.model.exponential.centre
    exponential.normalization = result.model.exponential.normalization
    exponential.rate = result.instance.exponential.rate

    model = af.Collection(gaussian=gaussian, exponential=exponential)

The ``model.info`` attribute shows that the parameter and prior passing has occurred on individual components.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    gaussian
        centre                         49.62558008533755
        normalization                  GaussianPrior, mean = 27.696267287676186, sigma = 13.848133643838093
        sigma                          10.13156736768096
    exponential
        centre                         GaussianPrior, mean = 50.31368408748868, sigma = 20.0
        normalization                  GaussianPrior, mean = 38.46597213618446, sigma = 19.23298606809223
        rate                           0.04928930602303

Take Attributes
---------------

The examples above linked models where the individual model components that were passed stayed the same.

We can link two related models, where only a subset of parameters are shared, by using the ``take_attributes()`` method.

For example, lets define a ``GaussianKurtosis`` which is a ``Gaussian`` with an extra parameter for its kurtosis.

.. code-block:: python

    class GaussianKurtosis:
        def __init__(
            self,
            centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
            normalization=1.0,  # <- are the Gaussian``s model parameters.
            sigma=5.0,
               kurtosis=1.0,
        ):
            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma
            self.kurtosis = kurtosis

The ``take_attributes()`` method takes a ``source`` model component, and inspects the names of all its parameters.

For  the ``Gaussian`` model result input below, it finds the parameters ``centre``, ``normalization`` and ``sigma``.

It then finds all parameters in the new ``model`` which have the same names, which for the ``GaussianKurtosis`` is
``centre``, ``normalization`` and ``sigma``.

For all parameters which have the same name, the parameter is passed.

.. code-block:: python

    model = af.Collection(gaussian=af.Model(GaussianKurtosis))
    model.kurtosis = af.Uniform(lower_limit=-1.0, upper_limit=1.0)

    model.gaussian.take_attributes(source=result.model.gaussian)

Because the result was passed using ``model`` we see the priors on the ``GaussianKurtosis`` ``centre``,
``normalization`` and ``sigma`` have been updated, whereas its ``kurtosis`` has not.

.. code-block:: python

    print("GaussianKurtosis Model Priors After Take Attributes via Model \n")
    print("centre = ", model.gaussian.centre)
    print("normalization = ", model.gaussian.normalization)
    print("sigma = ", model.gaussian.sigma)
    print("kurtosis = ", model.gaussian.kurtosis)

This gives the following output:

.. code-block:: bash

    GaussianKurtosis Model Priors After Take Attributes via Model

    centre =  GaussianPrior, mean = 49.71429699925852, sigma = 20.0
    normalization =  GaussianPrior, mean = 27.696267287676186, sigma = 13.848133643838093
    sigma =  GaussianPrior, mean = 10.162401454722103, sigma = 5.081200727361051
    kurtosis =  UniformPrior, lower_limit = 0.0, upper_limit = 100.0

If we pass ``result.instance`` to take_attributes the same name linking is used, however parameters are passed as
floats.

.. code-block:: python

    model = af.Collection(gaussian=af.Model(GaussianKurtosis))
    model.kurtosis = af.Uniform(lower_limit=-1.0, upper_limit=1.0)

    model.take_attributes(source=result.instance.gaussian)

    print("Gaussian Model Priors After Take Attributes via Instance \n")
    print("centre = ", model.gaussian.centre)
    print("normalization = ", model.gaussian.normalization)
    print("sigma = ", model.gaussian.sigma)
    print("kurtosis = ", model.gaussian.kurtosis)

This gives the following output:

.. code-block:: bash

    Gaussian Model Priors After Take Attributes via Instance

    centre =  49.62558008533755
    normalization =  27.75178980320016
    sigma =  10.13156736768096
    kurtosis =  UniformPrior, lower_limit = 0.0, upper_limit = 100.0

As Model
--------

A common problem is when we have an ``instance`` (e.g. from a previous fit where we fixed the parameters)
but now wish to make its parameters free parameters again.

Furthermore, we may want to do this for specific model components.

The ``as_model`` method allows us to do this. Below, we pass the entire result (e.g. both the ``gaussian``
and ``exponential`` components), however we pass the ``Gaussian`` class to ``as_model``, meaning that any model
component in the ``instance`` which is a ``Gaussian`` will be converted to a model with free parameters.

.. code-block:: python

    model = result.instance.as_model((af.ex.Gaussian,))

    print("Gaussian Model Priors After via As model \n")
    print("centre = ", model.gaussian.centre)
    print("normalization = ", model.gaussian.normalization)
    print("sigma = ", model.gaussian.sigma)
    print("centre = ", model.exponential.centre)
    print("normalization = ", model.exponential.normalization)
    print("rate= ", model.exponential.rate)

This gives the following output:

.. code-block:: bash

    Gaussian Model Priors After via as_model:

    centre =  UniformPrior, lower_limit = 0.0, upper_limit = 100.0
    normalization =  LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
    sigma =  UniformPrior, lower_limit = 0.0, upper_limit = 25.0
    centre =  50.3182477457215
    normalization =  38.31003537202189
    rate=  0.04928930602303

The ``as_model()`` method does not have too much utility for the simple model used in this cookbook.

However, for multi-level models with many components, it is a powerful tool to compose custom models.

.. code-block:: python

    class MultiLevelProfiles:
        def __init__(
            self,
            higher_level_centre=50.0,  # This is the centre of all Gaussians in this multi level component.
            profile_list=None,  # This will contain a list of model-components
        ):

            self.higher_level_centre = higher_level_centre

            self.profile_list = profile_list


    multi_level_0 = af.Model(
        MultiLevelProfiles, profile_list=[af.ex.Gaussian, af.ex.Exponential, af.ex.Gaussian]
    )

    multi_level_1 = af.Model(
        MultiLevelProfiles,
        profile_list=[af.ex.Gaussian, af.ex.Exponential, af.ex.Exponential],
    )

    model = af.Collection(multi_level_0=multi_level_0, multi_level_1=multi_level_1)

This means every ``Gaussian`` in the complex multi-level model above would  have parameters set via the result of our
model-fit, if the model above was fitted such that it was contained in the result.

.. code-block:: python

model = result.instance.as_model((af.ex.Gaussian,))

Prior Passing
-------------

A complete description of how priors are passed via model linking can be found at the following notebook:

https://github.com/Jammy2211/autofit_workspace/blob/release/notebooks/features/search_chaining.ipynb

