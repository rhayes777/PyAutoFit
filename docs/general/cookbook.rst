.. _cookbook:

Cookbook
========

This cookbook therefore provides an API reference for model composition, in particular:

- Creating individual model components via ``af.Model``.
- Creating collections of model components as a single model using ``af.Collection``.
- Creating multi-level models using hierarchies of Python classes.

Examples using different **PyAutoFit** API's for model composition are provided, which produce more concise and
readable code for different use-cases.

Python Class Template
---------------------

A model component is written as a Python class using the following format:

- The name of the class is the name of the model component, in this case, "Gaussian".

- The input arguments of the constructor are the parameters of the mode (here ``centre``, ``normalization`` and ``sigma``).

- The default values of the input arguments tell **PyAutoFit** whether a parameter is a single-valued ``float`` or a multi-valued ``tuple``.

.. code-block:: python

    class Gaussian:
        def __init__(
            self,
            centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
            normalization=1.0,  # <- are the Gaussian``s model parameters.
            sigma=5.0,
        ):
            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma

Model
-----

To instantiate a Python class as a model component using the ``af.Model`` object:

.. code-block:: python

    model = af.Model(Gaussian)

To overwrite the priors of one or more parameters from the default value assumed via configuration files:

.. code-block:: python

    model = af.Model(Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.normalization = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
    model.sigma = af.GaussianPrior(mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1e5)

To fix a free parameter to a specific value (reducing the dimensionality of parameter space by 1):

.. code-block:: python

    model = af.Model(Gaussian)
    model.centre = 0.0

To link two parameters together such they always assume the same value (reducing the dimensionality of parameter space by 1):

.. code-block:: python

    model = af.Model(Gaussian)
    model.centre = model.normalization

Offsets between linked parameters are also possible:

.. code-block:: python

    model = af.Model(Gaussian)
    model.centre = model.normalization - 1.0
    model.centre = model.normalization + model.sigma

Assertions remove regions of parameter space:

.. code-block:: python

    model = af.Model(Gaussian)
    model.add_assertion(model.sigma > 5.0)
    model.add_assertion(model.centre > model.normalization)

Model (Alternative API)
-----------------------

The overwriting of priors shown above can be achieved via the following alternative API:

.. code-block:: python

    model = af.Model(
        Gaussian,
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        normalization=af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4),
        sigma=af.GaussianPrior(mean=0.0, sigma=1.0),
    )

This API can also be used for fixing a parameter to a certain value:

.. code-block:: python

    model = af.Model(Gaussian, centre=0.0)

Collection
----------

To instantiate multiple Python classes into a combined model component using ``af.Collection`` and ``af.Model``:

.. code-block:: python

    gaussian_0 = af.Model(Gaussian)
    gaussian_1 = af.Model(Gaussian)

    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

By setting up each ``Model`` first the model can be customized using either of the ``af.Model`` API's shown above:

.. code-block:: python

    gaussian_0 = af.Model(Gaussian)
    gaussian_0.normalization = 1.0
    gaussian_0.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    gaussian_1 = af.Model(
        Gaussian,
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        normalization=af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4),
        sigma=af.GaussianPrior(mean=0.0, sigma=1.0),
    )

    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

Collection (Alternative API)
----------------------------

To create the ``Collection`` in one line of Python by not defining each ``Model`` beforehand:

.. code-block:: python

    model = af.Collection(gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian))

Using this API, the ``af.Model()`` command can be omitted altogether (**PyAutoFit** will automatically determine
the ``Gaussian`` python classes should be set up as ``Model``'s):

.. code-block:: python

    model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)

To customize a model using this API the name of the model subcomponents (e.g. ``gaussian_0`` and ``gaussian_1``) are used
to access and customize the parameters.

.. code-block:: python

    model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)

    model.gaussian_0.normalization = 1.0
    model.gaussian_0.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    model.gaussian_0.centre = model.gaussian_1.centre

    model.gaussian_1.add_assertion(model.gaussian_1.sigma > 5.0)
    model.gaussian_1.centre = model.gaussian_1.normalization - 1.0

Multi-level Models (Advanced)
-----------------------------

A multi-level model component is written as a Python class using the following format:

- The input arguments include one or more optional lists of Python classes that themselves are instantiated as model components.

- Addition parameters specific to the higher level of the model can be included in the constructor (in this example a parameter called the ``higher_level_parameter`` is used).

Like a normal model component, the name of the Python class is the name of the model component, input arguments are
the parameters of the model and default values tell **PyAutoFit** whether a parameter is a single-valued ``float`` or a
multi-valued ``tuple``.

.. code-block:: python

    class MultiLevelGaussians:

        def __init__(
            self,
            higher_level_parameter=1.0,
            gaussian_list=None,  # This will optionally contain a list of ``af.Model(Gaussian)``'s
        ):

            self.higher_level_parameter = higher_level_parameter

            self.gaussian_list = gaussian_list

This multi-level model is instantiated via the ``af.Model()`` command, which is passed one or more ``Gaussian`` components:

.. code-block:: python

    multi_level = af.Model(
        MultiLevelGaussians, gaussian_list=[af.Model(Gaussian), af.Model(Gaussian)]
    )

Again, if the ``af.Model()`` on the individual ``Gaussian``'s is omitted they are still created as model components:

.. code-block:: python

    multi_level = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

To customize the higher level parameters of a multi-level the usual ``Model`` API is used:

.. code-block:: python

    multi_level = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

    multi_level.higher_level_parameter = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

To customize a multi-level model instantiated via lists, each model component is accessed via its index:

.. code-block:: python

    multi_level = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

    multi_level.gaussian_list[0].centre = multi_level.gaussian_list[1].centre

Any combination of the API's shown above can be used for customizing this model:

.. code-block:: python

    gaussian_0 = af.Model(Gaussian)
    gaussian_1 = af.Model(Gaussian)

    gaussian_0.centre = gaussian_1.centre

    multi_level = af.Model(
        MultiLevelGaussians, gaussian_list=[gaussian_0, gaussian_1, af.Model(Gaussian)]
    )

    multi_level.higher_level_parameter = 1.0
    multi_level.gaussian_list[2].centre = multi_level.gaussian_list[1].centre

Multi-level Models (Alternative API)
------------------------------------

A multi-level model can be instantiated where each model sub-component is setup using a name (as opposed to a list).

This means no list input parameter is required in the Python class of the model component:

.. code-block:: python

    class MultiLevelGaussians:

        def __init__(self, higher_level_parameter=1.0):

            self.higher_level_parameter = higher_level_parameter

        multi_level = af.Model(MultiLevelGaussians, gaussian_0=Gaussian, gaussian_1=Gaussian)

Each model subcomponent can be customized using its name, analogous to the ``Collection`` API:

.. code-block:: python

    multi_level = af.Model(MultiLevelGaussians, gaussian_0=Gaussian, gaussian_1=Gaussian)

    multi_level.gaussian_0.centre = multi_level.gaussian_1.centre

Multi-level Model Collections
-----------------------------

Models, multi-level models and collections can be combined to compose models of high complexity:

.. code-block:: python

    multi_level_0 = af.Model(MultiLevelGaussians, gaussian_0=Gaussian, gaussian_1=Gaussian)

    multi_level_1 = af.Model(
        MultiLevelGaussians, gaussian_0=Gaussian, gaussian_1=Gaussian, gaussian_2=Gaussian
    )

    model = af.Collection(multi_level_0=multi_level_0, multi_level_1=multi_level_1)

    print(model.multi_level_0.gaussian_1.centre)
    print(model.multi_level_1.higher_level_parameter)

Wrap Up
-------

The API described here can be extended in all the ways one would expect.

For example, multi-level models composed of multiple levels are possible:

.. code-block:: python

    multi_level_x2_model = af.Model(
        MultiLevelGaussians,
        multi_level_0=af.Model(MultiLevelGaussians, gaussian_0=Gaussian),
        multi_level_1=af.Model(MultiLevelGaussians, gaussian_0=Gaussian),
    )

    print(multi_level_x2_model.multi_level_0.gaussian_0.centre)