.. _cookbook_2_collections:

Collections
===========

This cookbook provides an overview of the basic model composition tools, specifically the ``Collection`` object,
whuich groups together multiple ``Model()`` components in order to compose complex models.

Examples using different PyAutoFit API’s for model composition are provided, which produce more concise and readable
code for different use-cases.

Python Classes
--------------

We will use two model components, written using the **PyAutoFit** class template, to illustrate ``Collection``'s:

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


    class Exponential:
        def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
            normalization=0.1,  # <- parameters of the Exponential.
            rate=0.01,
        ):
            self.centre = centre
            self.normalization = normalization
            self.rate = rate

Model Composition
-----------------

To instantiate multiple Python classes into a combined model component we combine the ``af.Collection()`` and ``af.Model()``
objects.

By passing the key word arguments ``gaussian`` and ``exponential`` below, these are used as the names of the attributes of
instances created using this model (which is illustrated clearly below).

.. code-block:: python

    gaussian = af.Model(Gaussian)
    exponential = af.Model(Exponential)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

We can check the model has a ``prior_count`` of 6, meaning the 3 parameters defined above (``centre``, ``normalization``,
``sigma`` and ``rate``) for both the ``Gaussian`` and ``Exponential`` classes all have priors associated with them .

This also means each parameter is fitted for if we fitted the model to data via a non-linear search.

.. code-block:: python

    print(f"Model Prior Count = {model.prior_count}")

Printing the ``info`` attribute of the model gives us information on all of the parameters, their priors and the
structure of the model collection.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Model Prior Count = 6
    gaussian
        centre                         UniformPrior, lower_limit = 0.0, upper_limit = 100.0
        normalization                  LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                          UniformPrior, lower_limit = 0.0, upper_limit = 25.0
    exponential
        centre                         UniformPrior, lower_limit = 0.0, upper_limit = 100.0
        normalization                  LogUniformPrior, lower_limit = 1e-06, upper_limit = 1000000.0
        rate                           UniformPrior, lower_limit = 0.0, upper_limit = 1.0

Instances
---------

We can create an instance of collection containing both the ``Gaussian`` and ``Exponential`` classes using this model.

Below, we create an ``instance`` where:

- The ``Gaussian`` class has``centre=30.0``, ``normalization=2.0`` and ``sigma=3.0``.
- The ``Exponential`` class has``centre=60.0``, ``normalization=4.0`` and ``rate=1.0``.

.. code-block:: python

    instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0, 60.0, 4.0, 1.0])

Because we passed the key word arguments ``gaussian`` and ``exponential`` above, these are the names of the attributes of
instances created using this model (e.g. this is why we write ``instance.gaussian``):

.. code-block:: python

    print("Instance Parameters \n")
    print("centre (Gaussian) = ", instance.gaussian.centre)
    print("normalization (Gaussian)  = ", instance.gaussian.normalization)
    print("sigma (Gaussian)  = ", instance.gaussian.sigma)
    print("centre (Exponential) = ", instance.exponential.centre)
    print("normalization (Exponential) = ", instance.exponential.normalization)
    print("sigma (Exponential) = ", instance.exponential.rate)

This gives the following output:

.. code-block:: bash

    Instance Parameters

    centre (Gaussian) =  30.0
    normalization (Gaussian)  =  2.0
    sigma (Gaussian)  =  3.0
    centre (Exponential) =  60.0
    normalization (Exponential) =  4.0
    sigma (Exponential) =  1.0

Alternatively, the instance's variables can also be accessed as a list, whereby instead of using attribute names
(e.g. ``gaussian_0``) we input the list index.

Note that the order of the instance model components is derived by the order the components are input into the model.

For example, for the line ``af.Collection(gaussian=gaussian, exponential=exponential)``, the first entry in the list
is the gaussian because it is the first input to the ``Collection``.

.. code-block:: python

    print("centre (Gaussian) = ", instance[0].centre)
    print("normalization (Gaussian)  = ", instance[0].normalization)
    print("sigma (Gaussian)  = ", instance[0].sigma)
    print("centre (Gaussian) = ", instance[1].centre)
    print("normalization (Gaussian) = ", instance[1].normalization)
    print("sigma (Exponential) = ", instance[1].rate)

This gives the same output as before:

.. code-block:: bash

    Instance Parameters

    centre (Gaussian) =  30.0
    normalization (Gaussian)  =  2.0
    sigma (Gaussian)  =  3.0
    centre (Exponential) =  60.0
    normalization (Exponential) =  4.0
    sigma (Exponential) =  1.0

Implicit Model
--------------

When creating a model via a ``Collection``, there is no need to actually pass the python classes as an ``af.Model()``
because **PyAutoFit** implicitly assumes they are to be created as a ``Model()``..

This enables more concise code, whereby the following code:

.. code-block:: python

    gaussian = af.Model(Gaussian)
    exponential = af.Model(Exponential)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

Can instead be written as:

.. code-block:: python

    model = af.Collection(gaussian=Gaussian, exponential=Exponential)

Model Customization
-------------------

By setting up each Model first the model can be customized using either of the ``af.Model()`` API’s shown above:

.. code-block:: python

    gaussian = af.Model(Gaussian)
    gaussian.normalization = 1.0
    gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    exponential = af.Model(Exponential)
    exponential.centre = 50.0
    exponential.add_assertion(exponential.rate > 5.0)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

Below is an alternative API that can be used to create the same model as above:

.. code-block:: python

    gaussian = af.Model(
        Gaussian, normalization=1.0, sigma=af.GaussianPrior(mean=0.0, sigma=1.0)
    )
    exponential = af.Model(Exponential, centre=50.0)
    exponential.add_assertion(exponential.rate > 5.0)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

Model Customization After Collection
------------------------------------

After creating the model as a ``Collection`` we can customize it afterwards:

.. code-block:: python

    model = af.Collection(gaussian=Gaussian, exponential=Exponential)

    model.gaussian.normalization = 1.0
    model.gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    model.exponential.centre = 50.0
    model.exponential.add_assertion(exponential.rate > 5.0)

Many Components
---------------

There is no limit to the number of components we can use to set up a model via a ``Collection``.

.. code-block:: python

    model = af.Collection(
        gaussian_0=Gaussian,
        gaussian_1=Gaussian,
        exponential_0=Exponential,
        exponential_1=Exponential,
        exponential_2=Exponential,
    )

Model Composition via Dictionaries
----------------------------------

A model can be created via ``af.Collection()`` where a dictionary of ``af.Model()`` objects are passed to it.

The two models created below are identical - one uses the API detailed above whereas the second uses a dictionary.

.. code-block:: python

    model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)

    model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}
    model = af.Collection(**model_dict)

The keys of the dictionary passed to the model (e.g. ``gaussian_0`` and ``gaussian_1`` above) are used to create the
names of the attributes of instances of the model.

.. code-block:: python

    instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    print("Instance Parameters \n")
    print("centre (Gaussian) = ", instance.gaussian_0.centre)
    print("normalization (Gaussian)  = ", instance.gaussian_0.normalization)
    print("sigma (Gaussian)  = ", instance.gaussian_0.sigma)
    print("centre (Gaussian) = ", instance.gaussian_1.centre)
    print("normalization (Gaussian) = ", instance.gaussian_1.normalization)
    print("sigma (Gaussian) = ", instance.gaussian_1.sigma)

Model Composition via Lists
---------------------------

A list of model components can also be passed to an ``af.Collection`` to create a model:

.. code-block:: python

    model = af.Collection([Gaussian, Gaussian])

When a list is used, there is no string with which to name the model components (e.g. we do not input ``gaussian_0``
and ``gaussian_1`` anywhere.

The ``instance`` therefore can only be accessed via list indexing.

.. code-block:: python

    instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    print("Instance Parameters \n")
    print("centre (Gaussian) = ", instance[0].centre)
    print("normalization (Gaussian)  = ", instance[0].normalization)
    print("sigma (Gaussian)  = ", instance[0].sigma)
    print("centre (Gaussian) = ", instance[1].centre)
    print("normalization (Gaussian) = ", instance[1].normalization)
    print("sigma (Gaussian) = ", instance[1].sigma)

Model Dictionary
----------------

A ``Collection`` has a ``dict`` attribute, which express all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.

.. code-block:: python

    model = af.Model(Gaussian)

    print(model.dict())

Model Addition
--------------

If we have two `Collection()

JSon Outputs
------------

Python dictionaries can easily be saved to hard disk as a ``.json`` file.

This means we can save any **PyAutoFit** model to hard-disk, even when it is composed using ``Collection``'s:

.. code-block:: python

    model_path = path.join("path", "to", "jsons")

    os.makedirs(model_path, exist_ok=True)

    model_file = path.join(model_path, "collection.json")

    with open(model_file, "w+") as f:
        json.dump(model.dict(), f, indent=4)

We can load the model from its ``.json`` file.

This means in **PyAutoFit** one can easily writen a model, save it to hard disk and load it elsewhere.

.. code-block:: python

    model = af.Model.from_json(file=model_file)

Wrap Up
-------

This cookbook shows how to compose models consisting of multiple components using the ``af.Collection()`` object.

The next cookbook describes how **PyAutoFit**'s model composition tools can be used to customize models which
fit multiple datasets simultaneously.