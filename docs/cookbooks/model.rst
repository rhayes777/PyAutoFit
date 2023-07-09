.. _model:

Model
=====

Model composition is the process of defining a probabilistic model as a collection of model components, which are
ultimate fitted to a dataset via a non-linear search.

This cookbook provides an overview of basic model composition tools.

**Contents:**

If first describes how to use the ``af.Model`` object to define models with a single model component from single
Python classes, with the following sections:

- **Python Class Template**: The template of a model component written as a Python class.
- **Model Composition (Model)**: Creating a model via ``af.Model()``.
- **Priors (Model)**: How the default priors of a model are set and how to customize them.
- **Instances (Model)**: Creating an instance of a model via input parameters.
- **Model Customization (Model)**: Customizing a model (e.g. fixing parameters or linking them to one another).
- **Json Output (Model)**: Output a model in human readable text via a .json file and loading it back again.

It then describes how to use the ``af.Collection`` object to define models with many model components from multiple
Python classes, with the following sections:

- **Model Composition (Collection)**: Creating a model via ``af.Collection()``.
- **Priors (Collection)**: How the default priors of a collection are set and how to customize them.
- **Instances (Collection)**: Create an instance of a collection via input parameters.
- **Model Customization (Collection)**: Customize a collection (e.g. fixing parameters or linking them to one another).
- **Json Output (Collection)**: Output a collection in human readable text via a .json file and loading it back again.
- **Extensible Models (Collection)**: Using collections to extend models with new model components, including the use of Python dictionaries and lists.

Python Class Template
---------------------

A model component is written as a Python class using the following format:

- The name of the class is the name of the model component, in this case, “Gaussian”.

- The input arguments of the constructor are the parameters of the mode (here ``centre``, ``normalization`` and ``sigma``).

- The default values of the input arguments tell PyAutoFit whether a parameter is a single-valued float or a multi-valued tuple.

We define a 1D Gaussian model component to illustrate model composition in PyAutoFit.

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

Model Composition (Model)
-------------------------

We can instantiate a Python class as a model component using ``af.Model()``.

.. code-block:: python

    model = af.Model(Gaussian)

The model has 3 free parameters, corresponding to the 3 parameters defined above (``centre``, ``normalization``
and ``sigma``).

Each parameter has a prior associated with it, meaning they are fitted for if the model is passed to a non-linear 
search.

.. code-block:: python

    print(f"Model Total Free Parameters = {model.total_free_parameters}")

If we print the ``info`` attribute of the model we get information on all of the parameters and their priors.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 3

    model                         Gaussian (N=3)

    centre                        UniformPrior [1], lower_limit = 0.0, upper_limit = 100.0
    normalization                 LogUniformPrior [2], lower_limit = 1e-06, upper_limit = 1000000.0
    sigma                         UniformPrior [3], lower_limit = 0.0, upper_limit = 25.0

Priors (Model)
--------------

The model has a set of default priors, which have been loaded from a config file in the PyAutoFit workspace.

The config cookbook describes how to setup config files in order to produce custom priors, which means that you do not
need to manually specify priors in your Python code every time you compose a model.

If you do not setup config files, all priors must be manually specified before you fit the model, as shown below.

.. code-block:: python

    model = af.Model(Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.LogUniformPrior(lower_limit=1e-4, upper_limit=1e4)
    model.sigma = af.GaussianPrior(mean=0.0, sigma=1.0, lower_limit=0.0, upper_limit=1e5)

Instances (Model)
-----------------

Instances of the model components above (created via ``af.Model``) can be created, where an input ``vector`` of
parameters is mapped to create an instance of the Python class of the model.

We first need to know the order of parameters in the model, so we know how to define the input ``vector``. This
information is contained in the models ``paths`` attribute:

.. code-block:: python

    print(model.paths)

The paths appear as follows:

.. code-block:: bash

    [('centre',), ('normalization',), ('sigma',)]

We create an ``instance`` of the ``Gaussian`` class via the model where ``centre=30.0``, ``normalization=2.0`` and ``sigma=3.0``.

.. code-block:: python

    instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0])

    print("Model Instance: \n")
    print(instance)

    print("Instance Parameters \n")
    print("centre = ", instance.centre)
    print("normalization = ", instance.normalization)
    print("sigma = ", instance.sigma)

This gives the following output:

.. code-block:: bash

    Model Instance:
    <__main__.Gaussian object at 0x7f6f11d437c0>

    Instance Parameters

    centre =  30.0
    normalization =  2.0
    sigma =  3.0

We can create an ``instance`` by inputting unit values (e.g. between 0.0 and 1.0) which are mapped to the input values
via the priors.

The inputs of 0.5 below are mapped as follows:

- ``centre``: goes to 0.5 because this is the midpoint of a ``UniformPrior`` with ``lower_limit=0.0`` and ``upper_limit=1.0``.

- ``normalization`` goes to 1.0 because this is the midpoint of the ``LogUniformPrior``' with ``lower_limit=1e-4`` and ``upper_limit=1e4`` corresponding to log10 space.

- ``sigma``: goes to 0.0 because this is the ``mean`` of the ``GaussianPrior``.

.. code-block:: python

    instance = model.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.5])

    print("Model Instance:\n")
    print(instance)

    print("\nInstance Parameters \n")
    print("centre = ", instance.centre)
    print("normalization = ", instance.normalization)
    print("sigma = ", instance.sigma)

This gives the following output:

.. code-block:: bash

    Model Instance:
    <__main__.Gaussian object at 0x7f6f11d43f70>

    Instance Parameters

    centre =  50.0
    normalization =  1.0
    sigma =  0.0

We can create instances of the ``Gaussian`` using the median value of the prior of every parameter.

.. code-block:: python

    instance = model.instance_from_prior_medians()

    print("Instance Parameters \n")
    print("centre = ", instance.centre)
    print("normalization = ", instance.normalization)
    print("sigma = ", instance.sigma)

This gives the following output:

.. code-block:: bash

    Instance Parameters

    centre =  50.0
    normalization =  1.0
    sigma =  0.0

We can create a random instance, where the random values are unit values drawn between 0.0 and 1.0.

This means the parameter values of this instance are randomly drawn from the priors.

.. code-block:: python

    model = af.Model(Gaussian)
    instance = model.random_instance()

Model Customization (Model)
---------------------------

We can fix a free parameter to a specific value (reducing the dimensionality of parameter space by 1):

.. code-block:: python

    model = af.Model(Gaussian)
    model.centre = 0.0

We can link two parameters together such they always assume the same value (reducing the dimensionality of 
parameter space by 1):

.. code-block:: python

    model.centre = model.normalization

Offsets between linked parameters or with certain values are possible:

.. code-block:: python

    model.centre = model.normalization + model.sigma

Assertions remove regions of parameter space (but do not reduce the dimensionality of parameter space):

.. code-block:: python

    model.add_assertion(model.sigma > 5.0)
    model.add_assertion(model.centre > model.normalization)

The customized model can be inspected by printing its `info` attribute.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 2
    
    model                            Gaussian (N=2)
        centre                       SumPrior (N=2)
    
    centre
        self                         LogUniformPrior [14], lower_limit = 1e-06, upper_limit = 1000000.0
        other                        UniformPrior [15], lower_limit = 0.0, upper_limit = 25.0
    normalization                    LogUniformPrior [14], lower_limit = 1e-06, upper_limit = 1000000.0
    sigma                            UniformPrior [15], lower_limit = 0.0, upper_limit = 25.0

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


Json Outputs (Model)
--------------------

A model has a ``dict`` attribute, which expresses all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.

.. code-block:: python

    model = af.Model(Gaussian)

    print(model.dict())

This gives the following output:

.. code-block:: bash

    {
        'class_path': '__main__.Gaussian', 'type': 'model',
        'centre': {'lower_limit': 0.0, 'upper_limit': 100.0, 'type': 'Uniform'},
        'normalization': {'lower_limit': 1e-06, 'upper_limit': 1000000.0, 'type': 'LogUniform'},
        'sigma': {'lower_limit': 0.0, 'upper_limit': 25.0, 'type': 'Uniform'}
    }

The dictionary representation printed above can be saved to hard disk as a ``.json`` file.

This means we can save any **PyAutoFit** model to hard-disk in a human readable format.

Checkout the file ``autofit_workspace/*/cookbooks/jsons/model.json`` to see the model written as a .json.

.. code-block:: python

    model_path = path.join("scripts", "cookbooks", "jsons")

    os.makedirs(model_path, exist_ok=True)

    model_file = path.join(model_path, "model.json")

    with open(model_file, "w+") as f:
        json.dump(model.dict(), f, indent=4)

We can load the model from its ``.json`` file, meaning that one can easily save a model to hard disk and load it
elsewhere.

.. code-block:: python

    model = af.Model.from_json(file=model_file)

Model Composition (Collection)
------------------------------

To illustrate ``Collection`` objects we define a second model component, representing a ``Exponential`` profile.

.. code-block:: python

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

To instantiate multiple Python classes into a combined model component we combine the ``af.Collection()`` and ``af.Model()``
objects.

By passing the key word arguments ``gaussian`` and ``exponential`` below, these are used as the names of the attributes of
instances created using this model (which is illustrated clearly below).

.. code-block:: python

    model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

We can check the model has a ``total_free_parameters`` of 6, meaning the 3 parameters defined
above (``centre``, ``normalization``, ``sigma`` and ``rate``) for both the ``Gaussian`` and ``Exponential`` classes all have
priors associated with them .

This also means each parameter is fitted for if we fitted the model to data via a non-linear search.

.. code-block:: python

    print(f"Model Total Free Parameters = {model.total_free_parameters}")

Printing the ``info`` attribute of the model gives us information on all of the parameters.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 6

    model                         Collection (N=6)
        gaussian                  Gaussian (N=3)
        exponential               Exponential (N=3)

    gaussian
        centre                    UniformPrior [39], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [40], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [41], lower_limit = 0.0, upper_limit = 25.0
    exponential
        centre                    UniformPrior [42], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [43], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [44], lower_limit = 0.0, upper_limit = 1.0

Priors (Collection)
-------------------

The model has a set of default priors, which have been loaded from a config file in the PyAutoFit workspace.

The configs cookbook describes how to setup config files in order to produce custom priors, which means that you do not
need to manually specify priors in your Python code every time you compose a model.

If you do not setup config files, all priors must be manually specified before you fit the model, as shown below.

.. code-block:: python

    model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

When creating a model via a ``Collection``, there is no need to actually pass the python classes as an ``af.Model()``
because **PyAutoFit** implicitly assumes they are to be created as a ``Model()``.

This enables more concise code, whereby the following code:

.. code-block:: python

    model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

Can instead be written as:

.. code-block:: python

    model = af.Collection(gaussian=Gaussian, exponential=Exponential)

Instances (Collection)
----------------------

We can create an instance of collection containing both the ``Gaussian`` and ``Exponential`` classes using this model.

Below, we create an ``instance`` where:

- The ``Gaussian`` class has ``centre=30.0``, ``normalization=2.0`` and ``sigma=3.0``.
- The ``Exponential`` class has ``centre=60.0``, ``normalization=4.0`` and ``rate=1.0````.

.. code-block:: python

    instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0, 60.0, 4.0, 1.0])

Because we passed the key word arguments ``gaussian`` and ``exponential`` above, these are the names of the attributes of
instances created using this model (e.g. this is why we write ``instance.gaussian``):

.. code-block:: python

    print("Model Instance: \n")
    print(instance)

    print("Instance Parameters \n")
    print("centre (Gaussian) = ", instance.gaussian.centre)
    print("normalization (Gaussian)  = ", instance.gaussian.normalization)
    print("sigma (Gaussian)  = ", instance.gaussian.sigma)
    print("centre (Exponential) = ", instance.exponential.centre)
    print("normalization (Exponential) = ", instance.exponential.normalization)
    print("rate (Exponential) = ", instance.exponential.rate)

This gives the following output:

.. code-block:: bash

    Model Instance:
    <autofit.mapper.model.ModelInstance object at 0x7f6f11b73580>

    Instance Parameters

    centre (Gaussian) =  30.0
    normalization (Gaussian)  =  2.0
    sigma (Gaussian)  =  3.0
    centre (Exponential) =  60.0
    normalization (Exponential) = 4.0
    rate (Exponential) =  1.0

Alternatively, the instance's variables can also be accessed as a list, whereby instead of using attribute names
(e.g. ``gaussian_0``) we input the list index.

Note that the order of the instance model components is determined from the order the components are input into the 
``Collection``.

For example, for the line ``af.Collection(gaussian=gaussian, exponential=exponential)``, the first entry in the list
is the gaussian because it is the first input to the ``Collection``.

.. code-block:: python

    print("centre (Gaussian) = ", instance[0].centre)
    print("normalization (Gaussian)  = ", instance[0].normalization)
    print("sigma (Gaussian)  = ", instance[0].sigma)
    print("centre (Gaussian) = ", instance[1].centre)
    print("normalization (Gaussian) = ", instance[1].normalization)
    print("rate (Exponential) = ", instance[1].rate)

This gives the following output:

.. code-block:: bash

    centre (Gaussian) =  30.0
    normalization (Gaussian)  =  2.0
    sigma (Gaussian)  =  3.0
    centre (Exponential) =  60.0
    normalization (Exponential) = 4.0
    rate (Exponential) =  1.0

Model Customization (Collection)
--------------------------------

By setting up each Model first the model can be customized using either of the API’s shown above:

.. code-block:: python

    gaussian = af.Model(Gaussian)
    gaussian.normalization = 1.0
    gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    exponential = af.Model(Exponential)
    exponential.centre = 50.0
    exponential.add_assertion(exponential.rate > 5.0)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

    print(model.info)

This gives the following output:

.. code-block:: bash

Total Free Parameters = 4

    model                         Collection (N=4)
        gaussian                  Gaussian (N=2)
        exponential               Exponential (N=2)

    gaussian
        centre                    UniformPrior [71], lower_limit = 0.0, upper_limit = 100.0
        normalization             1.0
        sigma                     GaussianPrior [70], mean = 0.0, sigma = 1.0
    exponential
        centre                    50.0
        normalization             LogUniformPrior [72], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [73], lower_limit = 0.0, upper_limit = 1.0

Below is an alternative API that can be used to create the same model as above.

Which API is used is up to the user and which they find most intuitive.

.. code-block:: python

    gaussian = af.Model(
        Gaussian, normalization=1.0, sigma=af.GaussianPrior(mean=0.0, sigma=1.0)
    )
    exponential = af.Model(Exponential, centre=50.0)
    exponential.add_assertion(exponential.rate > 5.0)

    model = af.Collection(gaussian=gaussian, exponential=exponential)

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 4

    model                         Collection (N=4)
        gaussian                  Gaussian (N=2)
        exponential               Exponential (N=2)

    gaussian
        centre                    UniformPrior [63], lower_limit = 0.0, upper_limit = 100.0
        normalization             1.0
        sigma                     GaussianPrior [66], mean = 0.0, sigma = 1.0
    exponential
        centre                    50.0
        normalization             LogUniformPrior [68], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [69], lower_limit = 0.0, upper_limit = 1.0

After creating the model as a ``Collection`` we can customize it afterwards:

.. code-block:: python

    model = af.Collection(gaussian=Gaussian, exponential=Exponential)

    model.gaussian.normalization = 1.0
    model.gaussian.sigma = af.GaussianPrior(mean=0.0, sigma=1.0)

    model.exponential.centre = 50.0
    model.exponential.add_assertion(exponential.rate > 5.0)

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 4

    model                         Collection (N=4)
        gaussian                  Gaussian (N=2)
        exponential               Exponential (N=2)

    gaussian
        centre                    UniformPrior [71], lower_limit = 0.0, upper_limit = 100.0
        normalization             1.0
        sigma                     GaussianPrior [70], mean = 0.0, sigma = 1.0
    exponential
        centre                    50.0
        normalization             LogUniformPrior [72], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [73], lower_limit = 0.0, upper_limit = 1.0

JSon Outputs (Collection)
-------------------------

A ``Collection`` has a ``dict`` attribute, which express all information about the model as a Python dictionary.

By printing this dictionary we can therefore get a concise summary of the model.

.. code-block:: python

    model = af.Model(Gaussian)

    print(model.dict())

This gives the following output:

.. code-block:: bash

    {
        'type': 'collection',
        'gaussian': {
            'class_path': '__main__.Gaussian', 'type': 'model',
            'centre': {'lower_limit': 0.0, 'upper_limit': 100.0, 'type': 'Uniform'},
            'normalization': 1.0, 'sigma': {'lower_limit': -inf, 'upper_limit': inf, 'type': 'Gaussian', 'mean': 0.0, 'sigma': 1.0}},
        'exponential': {
            'class_path': '__main__.Exponential', 'type': 'model',
            'centre': 50.0,
            'normalization': {'lower_limit': 1e-06, 'upper_limit': 1000000.0, 'type': 'LogUniform'},
            'rate': {'lower_limit': 0.0, 'upper_limit': 1.0, 'type': 'Uniform'}}
    }

Python dictionaries can easily be saved to hard disk as a ``.json`` file.

This means we can save any **PyAutoFit** model to hard-disk.

Checkout the file ``autofit_workspace/*/model/jsons/model.json`` to see the model written as a .json.

.. code-block:: python

    model_path = path.join("scripts", "model", "jsons")

    os.makedirs(model_path, exist_ok=True)

    model_file = path.join(model_path, "collection.json")

    with open(model_file, "w+") as f:
        json.dump(model.dict(), f, indent=4)

We can load the model from its ``.json`` file, meaning that one can easily save a model to hard disk and load it
elsewhere.

.. code-block:: python

    model = af.Model.from_json(file=model_file)

    print(f"\n Model via Json Prior Count = {model.prior_count}")

Extensible Models (Collection)
------------------------------

There is no limit to the number of components we can use to set up a model via a ``Collection``.

.. code-block:: python

    model = af.Collection(
        gaussian_0=Gaussian,
        gaussian_1=Gaussian,
        exponential_0=Exponential,
        exponential_1=Exponential,
        exponential_2=Exponential,
    )

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 15

    model                         Collection (N=15)
        gaussian_0                Gaussian (N=3)
        gaussian_1                Gaussian (N=3)
        exponential_0             Exponential (N=3)
        exponential_1             Exponential (N=3)
        exponential_2             Exponential (N=3)

    gaussian_0
        centre                    UniformPrior [91], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [92], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [93], lower_limit = 0.0, upper_limit = 25.0
    gaussian_1
        centre                    UniformPrior [94], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [95], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [96], lower_limit = 0.0, upper_limit = 25.0
    exponential_0
        centre                    UniformPrior [97], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [98], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [99], lower_limit = 0.0, upper_limit = 1.0
    exponential_1
        centre                    UniformPrior [100], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [101], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [102], lower_limit = 0.0, upper_limit = 1.0
    exponential_2
        centre                    UniformPrior [103], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [104], lower_limit = 1e-06, upper_limit = 1000000.0
        rate                      UniformPrior [105], lower_limit = 0.0, upper_limit = 1.0
    Total Free Parameters = 6

    model                         Collection (N=6)
        gaussian_0                Gaussian (N=3)
        gaussian_1                Gaussian (N=3)

    gaussian_0
        centre                    UniformPrior [106], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [107], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [108], lower_limit = 0.0, upper_limit = 25.0
    gaussian_1
        centre                    UniformPrior [109], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [110], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [111], lower_limit = 0.0, upper_limit = 25.0
    Total Free Parameters = 6

    model                         Collection (N=6)
        gaussian_0                Gaussian (N=3)
        gaussian_1                Gaussian (N=3)

    gaussian_0
        centre                    UniformPrior [112], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [113], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [114], lower_limit = 0.0, upper_limit = 25.0
    gaussian_1
        centre                    UniformPrior [115], lower_limit = 0.0, upper_limit = 100.0
        normalization             LogUniformPrior [116], lower_limit = 1e-06, upper_limit = 1000000.0
        sigma                     UniformPrior [117], lower_limit = 0.0, upper_limit = 25.0

A model can be created via ``af.Collection()`` where a dictionary of ``af.Model()`` objects are passed to it.

The two models created below are identical- one uses the API detailed above whereas the second uses a dictionary.

.. code-block:: python

    model = af.Collection(gaussian_0=Gaussian, gaussian_1=Gaussian)

    model_dict = {"gaussian_0": Gaussian, "gaussian_1": Gaussian}
    model = af.Collection(**model_dict)


The keys of the dictionary passed to the model (e.g. ``gaussian_0`` and ``gaussian_1`` above) are used to create the
names of the attributes of instances of the model.

.. code-block:: python

    instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    print("Model Instance: \n")
    print(instance)

    print("Instance Parameters \n")
    print("centre (Gaussian) = ", instance.gaussian_0.centre)
    print("normalization (Gaussian)  = ", instance.gaussian_0.normalization)
    print("sigma (Gaussian)  = ", instance.gaussian_0.sigma)
    print("centre (Gaussian) = ", instance.gaussian_1.centre)
    print("normalization (Gaussian) = ", instance.gaussian_1.normalization)
    print("sigma (Gaussian) = ", instance.gaussian_1.sigma)


This gives the following output:

.. code-block:: bash

    Model Instance:
    <autofit.mapper.model.ModelInstance object at 0x7f10a40f3a60>

    Instance Parameters:

    centre (Gaussian) =  1.0
    normalization (Gaussian)  =  2.0
    sigma (Gaussian)  =  3.0
    centre (Gaussian) =  4.0
    normalization (Gaussian) =  5.0
    sigma (Gaussian) =  6.0

A list of model components can also be passed to an ``af.Collection`` to create a model:

.. code-block:: python

    model = af.Collection([Gaussian, Gaussian])

    print(model.info)

When a list is used, there is no string with which to name the model components (e.g. we do not input ``gaussian_0``
and ``gaussian_1`` anywhere.

The ``instance`` therefore can only be accessed via list indexing.

.. code-block:: python

    instance = model.instance_from_vector(vector=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    print("Model Instance: \n")
    print(instance)

    print("Instance Parameters \n")
    print("centre (Gaussian) = ", instance[0].centre)
    print("normalization (Gaussian)  = ", instance[0].normalization)
    print("sigma (Gaussian)  = ", instance[0].sigma)
    print("centre (Gaussian) = ", instance[1].centre)
    print("normalization (Gaussian) = ", instance[1].normalization)
    print("sigma (Gaussian) = ", instance[1].sigma)

This gives the following output:

.. code-block:: bash

    Model Instance:
    <autofit.mapper.model.ModelInstance object at 0x7f10a40f3a60>

    Instance Parameters:

    centre (Gaussian) =  1.0
    normalization (Gaussian)  =  2.0
    sigma (Gaussian)  =  3.0
    centre (Gaussian) =  4.0
    normalization (Gaussian) =  5.0
    sigma (Gaussian) =  6.0

Wrap Up
-------

This cookbook shows how to compose models consisting of multiple components using the ``af.Model()``
and ``af.Collection()`` object.

Advanced model composition uses multi-level models, which compose models from hierarchies of Python classes. This is
described in the multi-level model cookbook. 

