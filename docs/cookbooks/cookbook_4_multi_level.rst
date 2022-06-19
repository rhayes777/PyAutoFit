.. _cookbook_4_multi_level

Multi Level Models
==================

A multi-level model component is written as a Python class where input arguments include one or more optional lists of
Python classes that themselves are instantiated as model components.

For example, the multi-level model below is a Python class that consists of a collection of 1D Gaussian's but has
all of their centres as its own higher level parameter:

.. code-block:: python

    class Gaussian:
        def __init__(
            self,
            normalization=1.0,  # <- **PyAutoFit** recognises these constructor arguments
            sigma=5.0,  # <- are the Gaussian``s model parameters.
        ):

            self.normalization = normalization
            self.sigma = sigma


    class MultiLevelGaussians:
        def __init__(
            self,
            higher_level_centre=50.0,  # This is the centre of all Gaussians in this multi level component.
            gaussian_list=None,  # This will contain a list of ``af.Model(Gaussian)``'s
        ):

            self.higher_level_centre = higher_level_centre

            self.gaussian_list = gaussian_list

Composition
-----------

The multi-level model is instantiated via the af.Model() command, which is passed one or more Gaussian components:

.. code-block:: python

    model = af.Model(
        MultiLevelGaussians, gaussian_list=[af.Model(Gaussian), af.Model(Gaussian)]
    )

The multi-level model consists of two ``Gaussian``'s, however their centres are now shared as a high level parameter.

Thus, the total number of parameters is N=5 (x2 ``normalizations``, ``x2 ``sigma``'s and x1 ``higher_level_centre``.

Printing the ``info`` attribute of the model gives us information on all of the parameters, their priors and the
structure of the multi level model.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Multi-level Model Prior Count = 5
    higher_level_centre                UniformPrior, lower_limit = 0.0, upper_limit = 100.0
    gaussian_list
        0
            normalization              LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                      UniformPrior, lower_limit = 0.0, upper_limit = 25.0
        1
            normalization              LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                      UniformPrior, lower_limit = 0.0, upper_limit = 25.0

Instances
---------

When we create an instance via a multi-level model.

Its attributes are structured in a slightly different way to the ``Collection`` seen in previous cookbooks.

.. code-block:: python

    instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0])

    print("Instance Parameters \n")
    print("Normalization (Gaussian 0) = ", instance.gaussian_list[0].normalization)
    print("Sigma (Gaussian 0) = ", instance.gaussian_list[0].sigma)
    print("Normalization (Gaussian 0) = ", instance.gaussian_list[1].normalization)
    print("Sigma (Gaussian 0) = ", instance.gaussian_list[1].sigma)
    print("Higher Level Centre= ", instance.higher_level_centre)

This gives the following output:

.. code-block:: bash

    Instance Parameters

    Normalization (Gaussian 0) =  1.0
    Sigma (Gaussian 0) =  2.0
    Normalization (Gaussian 0) =  3.0
    Sigma (Gaussian 0) =  4.0
    Higher Level Centre=  5.0

Collection Equivalent
---------------------

An identical model in terms of functionality could of been created via the ``Collection`` object as follows:

.. code-block:: python

    class GaussianCentre:
        def __init__(
            self,
            centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
            normalization=1.0,  # <- are the Gaussian``s model parameters.
            sigma=5.0,
        ):
            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma


    model = af.Collection(gaussian_0=GaussianCentre, gaussian_1=GaussianCentre)

    model.gaussian_0.centre = model.gaussian_1.centre

When to Use a Multi Level Model?
--------------------------------

This raises the question of when to use a ``Collection`` and when to use multi-level models.

The answer depends on the structure of the models you are composing and fitting. It is common for many models to
have a natural multi-level structure.

For example, imagine we had a dataset with 3 groups of 1D ``Gaussian``'s with shared centres, where each group had 3
``Gaussian``'s.

This model is concise and easy to define using the multi-level API:

.. code-block:: python

    multi_0 = af.Model(MultiLevelGaussians, gaussian_list=3*[Gaussian])

    multi_1 = af.Model(MultiLevelGaussians, gaussian_list=3*[Gaussian])

    multi_2 = af.Model(MultiLevelGaussians, gaussian_list=3*[Gaussian])

    model = af.Collection(multi_0=multi_0, multi_1=multi_1, multi_2=multi_2)

Composing the same model without the multi-level model is less concise, less readable and prone to error:

.. code-block:: python

    multi_0 = af.Collection(
        gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
    )

    multi_0.gaussian_0.centre = multi_0.gaussian_1.centre
    multi_0.gaussian_0.centre = multi_0.gaussian_2.centre
    multi_0.gaussian_1.centre = multi_0.gaussian_2.centre

    multi_1 = af.Collection(
        gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
    )

    multi_1.gaussian_0.centre = multi_1.gaussian_1.centre
    multi_1.gaussian_0.centre = multi_1.gaussian_2.centre
    multi_1.gaussian_1.centre = multi_1.gaussian_2.centre

    multi_2 = af.Collection(
        gaussian_0=GaussianCentre, gaussian_1=GaussianCentre, gaussian_2=GaussianCentre
    )

    multi_2.gaussian_0.centre = multi_2.gaussian_1.centre
    multi_2.gaussian_0.centre = multi_2.gaussian_2.centre
    multi_2.gaussian_1.centre = multi_2.gaussian_2.centre

    model = af.Collection(multi_0=multi_0, multi_1=multi_1, multi_2=multi_2)

The multi-level model API is more **extensible**.

For example, if I wanted to compose a model with more ``Gaussians``, ``Exponential``'s and other 1D profiles I would simply
write:

.. code-block:: python

    multi = af.Model(
        MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian, Exponential, YourProfileHere]
    )

Composing the same model using just a ``Model`` and ``Collection`` is again possible, but would be even more cumbersome,
less readable and is not an API that is anywhere near as extensible as the multi-level model API.

Multi Level Model Customization
-------------------------------

To customize the higher level parameters of a multi-level the usual Model API is used:

.. code-block:: python

    multi = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

    multi.higher_level_centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

To customize a multi-level model instantiated via lists, each model component is accessed via its index:

.. code-block:: python

    multi = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

    multi_level = af.Model(MultiLevelGaussians, gaussian_list=[Gaussian, Gaussian])

    multi_level.gaussian_list[0].normalization = multi_level.gaussian_list[1].normalization

Any combination of the API’s shown above can be used for customizing this model:

.. code-block:: python

    gaussian_0 = af.Model(Gaussian)
    gaussian_1 = af.Model(Gaussian)

    gaussian_0.normalization = gaussian_1.normalization

    multi_level = af.Model(
        MultiLevelGaussians, gaussian_list=[gaussian_0, gaussian_1, af.Model(Gaussian)]
    )

    multi_level.higher_level_centre = 1.0
    multi_level.gaussian_list[2].normalization = multi_level.gaussian_list[1].normalization

Alternative API
---------------

A multi-level model can be instantiated where each model sub-component is setup using a name (as opposed to a list).

This means no list input parameter is required in the Python class of the model component, but we do need to include
the ``**kwargs`` input.

.. code-block:: python

    class MultiLevelGaussians:
        def __init__(self, higher_level_centre=1.0, **kwargs):

            self.higher_level_centre = higher_level_centre

    model = af.Model(
        MultiLevelGaussians, gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
    )

    print("Instance Parameters \n")
    print("Normalization (Gaussian 0) = ", instance.gaussian_0.normalization)
    print("Sigma (Gaussian 0) = ", instance.gaussian_0.sigma)
    print("Normalization (Gaussian 0) = ", instance.gaussian_1.normalization)
    print("Sigma (Gaussian 0) = ", instance.gaussian_1.sigma)
    print("Higher Level Centre= ", instance.higher_level_centre)

The use of Python dictionaries illustrated in previous cookbooks can also be used with multi-level models.

.. code-block:: python

    model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}

    model = af.Model(MultiLevelGaussians, **model_dict)

Model Dictionary
----------------

Multi level models also have a ``dict`` attribute, which express all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.

.. code-block:: python

    model = af.Model(Gaussian)

    print(model.dict())

JSon Outputs
------------

Python dictionaries can easily be saved to hard disk as a ``.json`` file.

This means we can save any **PyAutoFit** model to hard-disk:

.. code-block:: python

    model_path = path.join("path", "to", "jsons")

    os.makedirs(model_path, exist_ok=True)

    model_file = path.join(model_path, "multi_level.json")

    with open(model_file, "w+") as f:
        json.dump(model.dict(), f, indent=4)

We can load the model from its ``.json`` file.

This means in **PyAutoFit** one can easily writen a model, save it to hard disk and load it elsewhere.

.. code-block:: python

    model = af.Model.from_json(file=model_file)

Wrap Up
-------

This cookbook shows how to compose multi-level models from hierarchies of Python classes.

This is a compelling means by which to compose concise, readable and extendable models, if your modeling problem is
multi-level in its structure.