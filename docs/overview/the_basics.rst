.. _model_fit:

The Basics
==========

To illustrate **PyAutoFit** we'll use the example modeling problem of fitting a 1D Gaussian profile to noisy data.

To begin, lets import ``autofit`` (and ``numpy``) using the convention below:

.. code-block:: python

    import autofit as af
    import numpy as np

Example
-------

The example ``data`` with errors (black) and the model-fit (red), are shown below:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/data.png
  :width: 600
  :alt: Alternative text

The 1D signal was generated using a 1D Gaussian profile of the form:

.. math::

    g(x, I, \sigma) = \frac{N}{\sigma\sqrt{2\pi}} \exp{(-0.5 (x / \sigma)^2)}

Where:

``x``: Is the x-axis coordinate where the ``Gaussian`` is evaluated.

``N``: Describes the overall normalization of the Gaussian.

``sigma``: Describes the size of the Gaussian (Full Width Half Maximum = $\mathrm {FWHM}$ = $2{\sqrt {2\ln 2}}\;\sigma$)

Our modeling task is to fit the signal with a 1D Gaussian and recover its parameters (``x``, ``N``, ``sigma``).

Model
-----

We therefore need to define a 1D Gaussian as a "model component" in **PyAutoFit**.

A model component is written as a Python class using the following format:

- The name of the class is the name of the model component, in this case, "Gaussian".

- The input arguments of the constructor (the ``__init__`` method) are the parameters of the model, in this case ``centre``, ``normalization`` and ``sigma``.
  
- The default values of the input arguments define whether a parameter is a single-valued ``float`` or a multi-valued ``tuple``. In this case, all 3 input parameters are floats.
  
- It includes functions associated with that model component, which will be used when fitting the model to data.

.. code-block:: python
    class Gaussian:
        def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments
            normalization=0.1,  # <- are the Gaussian``s model parameters.
            sigma=0.01,
        ):
            """
            Represents a 1D ``Gaussian`` profile, which can be treated as a
            PyAutoFit model-component whose free parameters (centre,
            normalization and sigma) are fitted for by a non-linear search.

            Parameters
            ----------
            centre
                The x coordinate of the profile centre.
            normalization
                Overall normalization of the ``Gaussian`` profile.
            sigma
                The sigma value controlling the size of the Gaussian.
            """
            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma

        def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray) -> np.ndarray:
            """
            Returns the 1D Gaussian profile on a line of Cartesian x coordinates.

            The input xvalues are translated to a coordinate system centred on the
            Gaussian, by subtracting its centre.

            The output is referred to as the ``model_data`` to signify that it is
            a representation of the data from the model.

            Parameters
            ----------
            xvalues
                The x coordinates for which the Gaussian is evaluated.
            """
            transformed_xvalues = xvalues - self.centre

            return np.multiply(
                np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
                np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
            )


To compose a model using the ``Gaussian`` class above we use the ``af.Model`` object.

.. code-block:: python

    model = af.Model(Gaussian)
    print("Model ``Gaussian`` object: \n")
    print(model)

This gives the following output:

.. code-block:: bash

    Model `Gaussian` object:

    Gaussian (centre, UniformPrior [1], lower_limit = 0.0, upper_limit = 100.0),
    (normalization, LogUniformPrior [2], lower_limit = 1e-06, upper_limit = 1000000.0),
    (sigma, UniformPrior [3], lower_limit = 0.0, upper_limit = 25.0)

The model has a total of 3 parameters:

.. code-block:: python

    print(model.total_free_parameters)

All model information is given by printing its ``info`` attribute.

This shows that each model parameter has an associated prior.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 3

    model                                   Gaussian (N=3)

    centre                                  UniformPrior [1], lower_limit = 0.0, upper_limit = 100.0
    normalization                           LogUniformPrior [2], lower_limit = 1e-06, upper_limit = 1000000.0
    sigma                                   UniformPrior [3], lower_limit = 0.0, upper_limit = 25.0


The priors can be manually altered as follows, noting that these updated files will be used below when we fit the
model to data.

.. code-block:: python

    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)


Printing the ``model.info`` displayed these updated priors.

.. code-block:: python

    print(model.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 3

    model                                   Gaussian (N=3)

    centre                                  UniformPrior [4], lower_limit = 0.0, upper_limit = 100.0
    normalization                           UniformPrior [5], lower_limit = 0.0, upper_limit = 100.0
    sigma                                   UniformPrior [6], lower_limit = 0.0, upper_limit = 30.0

Instances
---------

Instances of the model components above (created via ``af.Model``) can be created, where an input ``vector`` of
parameters is mapped to create an instance of the Python class of the model.

We first need to know the order of parameters in the model, so we know how to define the input ``vector``. This
information is contained in the models ``paths`` attribute:

.. code-block:: python

    print(model.paths)

This gives the following output:

.. code-block:: bash

    [('centre',), ('normalization',), ('sigma',)]

We input values for the 3 free parameters of our model following the order of paths above:
 
1) ``centre=30.0``
2) ``normalization=2.0``
3) ``sigma=3.0``
 
This creates an ``instance`` of the Gaussian class via the model. 

.. code-block:: python

    instance = model.instance_from_vector(vector=[30.0, 2.0, 3.0])

This is an instance of the ``Gaussian`` class.

.. code-block:: python

    print("Model Instance: \n")
    print(instance)

This gives the following output:

.. code-block:: bash

    Model Instance:

    <__main__.Gaussian object at 0x7f3e37cb1990>

It has the parameters of the ``Gaussian`` with the values input above.

.. code-block:: python

    print("Instance Parameters \n")
    print("x = ", instance.centre)
    print("normalization = ", instance.normalization)
    print("sigma = ", instance.sigma)

This gives the following output:

.. code-block:: bash

    Instance Parameters

    x =  30.0
    normalization =  2.0
    sigma =  3.0

We can use functions associated with the class, specifically the ``model_data_1d_via_xvalues_from`` function, to 
create a realization of the ``Gaussian`` and plot it.

.. code-block:: python

    xvalues = np.arange(0.0, 100.0, 1.0)

    model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

    plt.plot(xvalues, model_data, color="r")
    plt.title("1D Gaussian Model Data.")
    plt.xlabel("x values of profile")
    plt.ylabel("Gaussian Value")
    plt.show()
    plt.clf()

Here is what the plot looks like:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/model_gaussian.png
  :width: 600
  :alt: Alternative text

This "model mapping", whereby models map to an instances of their Python classes, is integral to the core **PyAutoFit**
API for model composition and fitting.

Analysis
--------

Now we've defined our model, we need to inform **PyAutoFit** how to fit it to data.

We therefore define an ``Analysis`` class, which includes:

- An ``__init__`` constructor, which takes as input the ``data`` and ``noise_map``. This could be extended to include anything else necessary to fit the model to the data.

- A ``log_likelihood_function``, which defines how given an ``instance`` of the model we fit it to the data and return a log likelihood value.

Read the comments and docstrings of the ``Analysis`` object below in detail for more insights into how this object
works.

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(self, data: np.ndarray, noise_map: np.ndarray):
            """
            The ``Analysis`` class acts as an interface between the data and
            model in **PyAutoFit**.

            Its ``log_likelihood_function`` defines how the model is fitted to
            the data and it is called many times by the non-linear search fitting
            algorithm.

            In this example the ``Analysis`` ``__init__`` constructor only contains
            the ``data`` and ``noise-map``, but it can be easily extended to
            include other quantities.

            Parameters
            ----------
            data
                A 1D numpy array containing the data (e.g. a noisy 1D signal) f
                itted in the workspace examples.
            noise_map
                A 1D numpy array containing the noise values of the data, used
                for computing the goodness of fit metric, the log likelihood.
            """

            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance) -> float:
            """
            Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

            The data is fitted using an ``instance`` of the ``Gaussian`` class
            where its ``model_data_1d_via_xvalues_from`` is called in order to
            create a model data representation of the Gaussian that is fitted to the data.
            """

            """
            The ``instance`` that comes into this method is an instance of the ``Gaussian``
            model above, which was created via ``af.Model()``.

            The parameter values are chosen by the non-linear search, based on where
            it thinks the high likelihood regions of parameter space are.

            The lines of Python code are commented out below to prevent excessive print
            statements when we run the non-linear search, but feel free to uncomment
            them and run the search to see the parameters of every instance
            that it fits.


            # print("Gaussian Instance:")
            # print("Centre = ", instance.centre)
            # print("Normalization = ", instance.normalization)
            # print("Sigma = ", instance.sigma)
            """

            """
            Get the range of x-values the data is defined on, to evaluate the model of the Gaussian.
            """
            xvalues = np.arange(self.data.shape[0])

            """
            Use these xvalues to create model data of our Gaussian.
            """
            model_data = instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

            """
            Fit the model gaussian line data to the observed data, computing the residuals,
            chi-squared and log likelihood.
            """
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            chi_squared = sum(chi_squared_map)
            noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))
            log_likelihood = -0.5 * (chi_squared + noise_normalization)

            return log_likelihood


Create an instance of the ``Analysis`` class by passing the ``data`` and ``noise_map``.

.. code-block:: python

    analysis = Analysis(data=data, noise_map=noise_map)


Non Linear Search
-----------------

We have defined the model that we want to fit the data, and the analysis class that performs this fit.

We now choose our fitting algorithm, called the "non-linear search", and fit the model to the data.

For this example, we choose the nested sampling algorithm Dynesty. A wide variety of non-linear searches are 
available in **PyAutoFit** (see ?).

.. code-block:: python

    search = af.DynestyStatic(
        nlive=100,
        number_of_cores=1,
    )

Model Fit
---------

We begin the non-linear search by calling its ``fit`` method. 

.. code-block:: python

    print(
        The non-linear search has begun running.
        This Jupyter notebook cell with progress once the search
        has completed - this could take a few minutes!
    )

    result = search.fit(model=model, analysis=analysis)

    print("The search has finished run - you may now continue the notebook.")


Result
------

The result object returned by the fit provides information on the results of the non-linear search. 

The ``info`` attribute shows the result in a readable format.

.. code-block:: python

    print(result.info)

The output is as follows:

.. code-block:: bash

    Bayesian Evidence                       167.54413502
    Maximum Log Likelihood                  183.29775793
    Maximum Log Posterior                   183.29775793

    model                                   Gaussian (N=3)

    Maximum Log Likelihood Model:

    centre                                  49.880
    normalization                           24.802
    sigma                                   9.849


    Summary (3.0 sigma limits):

    centre                                  49.88 (49.51, 50.29)
    normalization                           24.80 (23.98, 25.67)
    sigma                                   9.84 (9.47, 10.25)


    Summary (1.0 sigma limits):

    centre                                  49.88 (49.75, 50.01)
    normalization                           24.80 (24.54, 25.11)
    sigma                                   9.84 (9.73, 9.97)

Results are returned as instances of the model, as we illustrated above in the model mapping section.

For example, we can print the result's maximum likelihood instance.

.. code-block:: python

    print(result.max_log_likelihood_instance)

    print("\nModel-fit Max Log-likelihood Parameter Estimates: \n")
    print("Centre = ", result.max_log_likelihood_instance.centre)
    print("Normalization = ", result.max_log_likelihood_instance.normalization)
    print("Sigma = ", result.max_log_likelihood_instance.sigma)

This gives the following output:

.. code-block:: bash

    Model-fit Max Log-likelihood Parameter Estimates:

    Centre =  49.87954357347897
    Normalization =  24.80227227310798
    Sigma =  9.84888033338011

A benefit of the result being an instance is that we can use any of its methods to inspect the results.

Below, we use the maximum likelihood instance to compare the maximum likelihood ``Gaussian`` to the data.

.. code-block:: python

    model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
        xvalues=np.arange(data.shape[0])
    )

    plt.errorbar(
        x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
    )
    plt.plot(xvalues, model_data, color="r")
    plt.title("Dynesty model fit to 1D Gaussian dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()

The plot appears as follows:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/toy_model_fit.png
  :width: 600
  :alt: Alternative text

Samples
-------

The results object also contains a ``Samples`` object, which contains all information on the non-linear search.

This includes parameter samples, log likelihood values, posterior information and results internal to the specific
algorithm (e.g. the internal dynesty samples).

This is described fully in the results overview, below we use the samples to plot the probability density function
cornerplot of the results.

.. code-block:: python

    search_plotter = aplt.DynestyPlotter(samples=result.samples)
    search_plotter.cornerplot()

The plot appears as follows:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/cornerplot.png
  :width: 600
  :alt: Alternative text

Extending Models
----------------

The model composition API is designed to make composing complex models, consisting of multiple components with many
free parameters, straightforward and scalable.

To illustrate this, we will extend our model to include a second component, representing a symmetric 1D Exponential
profile, and fit it to data generated with both profiles.

Lets begin by loading and plotting this data.

.. code-block:: python

    dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )
    xvalues = range(data.shape[0])
    plt.errorbar(
        x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
    )
    plt.show()
    plt.close()

The data appear as follows:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/data_2.png
  :width: 600
  :alt: Alternative text

We define a Python class for the ``Exponential`` model component, exactly as we did for the ``Gaussian`` above.

.. code-block:: python

    class Exponential:
        def __init__(
            self,
            centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
            normalization=1.0,  # <- are the Exponentials``s model parameters.
            rate=0.01,
        ):
            """
            Represents a symmetric 1D Exponential profile.

            Parameters
            ----------
            centre
                The x coordinate of the profile centre.
            normalization
                Overall normalization of the profile.
            ratw
                The decay rate controlling has fast the Exponential declines.
            """

            self.centre = centre
            self.normalization = normalization
            self.rate = rate

        def model_data_1d_via_xvalues_from(self, xvalues: np.ndarray):
            """
            Returns the symmetric 1D Exponential on an input list of Cartesian
            x coordinates.

            The input xvalues are translated to a coordinate system centred on
            the Exponential, via its ``centre``.

            The output is referred to as the ``model_data`` to signify that it
            is a representation of the data from the
            model.

            Parameters
            ----------
            xvalues
                The x coordinates in the original reference frame of the data.
            """

            transformed_xvalues = np.subtract(xvalues, self.centre)
            return self.normalization * np.multiply(
                self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
            )


We can easily compose a model consisting of 1 ``Gaussian`` object and 1 ``Exponential`` object using the ``af.Collection``
object:

.. code-block:: python

    model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

A ``Collection`` behaves analogous to a ``Model``, but it contains a multiple model components.

We can see this by printing its ``paths`` attribute, where paths to all 6 free parameters via both model components
are shown.

The paths have the entries ``.gaussian.`` and ``.exponential.``, which correspond to the names we input into  
the ``af.Collection`` above. 

.. code-block:: python

    print(model.paths)

The output is as follows:

.. code-block:: bash

    [
        ('gaussian', 'centre'),
        ('gaussian', 'normalization'),
        ('gaussian', 'sigma'),
        ('exponential', 'centre'),
        ('exponential', 'normalization'),
        ('exponential', 'rate')
    ]

We can use the paths to customize the priors of each parameter.

.. code-block:: python

    model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    model.exponential.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.exponential.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.exponential.rate = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

All of the information about the model created via the collection can be printed at once using its ``info`` attribute:

.. code-block:: python

    print(model.info)

The output appears as follows:

.. code-block:: bash

    Total Free Parameters = 6
    model                                       Collection (N=6)
            gaussian                            Gaussian (N=3)
            exponential                         Exponential (N=3)
        
        gaussian
            centre                              UniformPrior [13], lower_limit = 0.0, upper_limit = 100.0
            normalization                       UniformPrior [14], lower_limit = 0.0, upper_limit = 100.0
            sigma                               UniformPrior [15], lower_limit = 0.0, upper_limit = 30.0
        exponential
            centre                              UniformPrior [16], lower_limit = 0.0, upper_limit = 100.0
            normalization                       UniformPrior [17], lower_limit = 0.0, upper_limit = 100.0
            rate                                UniformPrior [18], lower_limit = 0.0, upper_limit = 10.0
    

A model instance can again be created by mapping an input ``vector``, which now has 6 entries.

.. code-block:: python

    instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])

This ``instance`` contains each of the model components we defined above. 

The argument names input into the ``Collection`` define the attribute names of the ``instance``:

.. code-block:: python

    print("Instance Parameters \n")
    print("x (Gaussian) = ", instance.gaussian.centre)
    print("normalization (Gaussian) = ", instance.gaussian.normalization)
    print("sigma (Gaussian) = ", instance.gaussian.sigma)
    print("x (Exponential) = ", instance.exponential.centre)
    print("normalization (Exponential) = ", instance.exponential.normalization)
    print("sigma (Exponential) = ", instance.exponential.rate)

The output appear as follows:

.. code-block:: bash

The ``Analysis`` class above assumed the ``instance`` contained only a single model-component.

We update its ``log_likelihood_function`` to use both model components in the ``instance`` to fit the data.

.. code-block:: python

    class Analysis(af.Analysis):
        def __init__(self, data: np.ndarray, noise_map: np.ndarray):
            """
            The ``Analysis`` class acts as an interface between the data and
            model in **PyAutoFit**.

            Its ``log_likelihood_function`` defines how the model is fitted to
            the data and it is called many times by the non-linear search
            fitting algorithm.

            In this example the ``Analysis`` ``__init__`` constructor only
            contains the ``data`` and ``noise-map``, but it can be easily
            extended to include other quantities.

            Parameters
            ----------
            data
                A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
            noise_map
                A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
                metric, the log likelihood.
            """

            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance) -> float:
            """
            Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

            The data is fitted using an ``instance`` of multiple 1D profiles
            (e.g. a ``Gaussian``, ``Exponential``) where
            their ``model_data_1d_via_xvalues_from`` methods are called and summed
            in order to create a model data representation that is fitted to the data.
            """

            """
            The ``instance`` that comes into this method is an instance of the
            ``Gaussian`` and ``Exponential`` models above, which were created
            via ``af.Collection()``.

            It contains instances of every class we instantiated it with, where
            each instance is named following the names given to the Collection,
            which in this example is a ``Gaussian`` (with name ``gaussian) and
            Exponential (with name ``exponential``).

            The parameter values are again chosen by the non-linear search,
            based on where it thinks the high likelihood regions of parameter
            space are. The lines of Python code are commented out below to
            prevent excessive print statements.


            # print("Gaussian Instance:")
            # print("Centre = ", instance.gaussian.centre)
            # print("Normalization = ", instance.gaussian.normalization)
            # print("Sigma = ", instance.gaussian.sigma)

            # print("Exponential Instance:")
            # print("Centre = ", instance.exponential.centre)
            # print("Normalization = ", instance.exponential.normalization)
            # print("Rate = ", instance.exponential.rate)
            """
            """
            Get the range of x-values the data is defined on, to evaluate
            the model of the Gaussian.
            """
            xvalues = np.arange(self.data.shape[0])

            """
            Internally, the ``instance`` variable is a list of all model
            omponents pass to the ``Collection`` above.

            we can therefore iterate over them and use their
            ``model_data_1d_via_xvalues_from`` methods to create the
            summed overall model data.
            """
            model_data = sum(
                [
                    profile_1d.model_data_1d_via_xvalues_from(xvalues=xvalues)
                    for profile_1d in instance
                ]
            )

            """
            Fit the model gaussian line data to the observed data, computing the residuals, chi-squared and log likelihood.
            """
            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            chi_squared = sum(chi_squared_map)
            noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
            log_likelihood = -0.5 * (chi_squared + noise_normalization)

            return log_likelihood



We can now fit this model to the data using the same API we did before.

.. code-block:: python

    analysis = Analysis(data=data, noise_map=noise_map)

    search = af.DynestyStatic(
        nlive=100,
        number_of_cores=1,
    )

    result = search.fit(model=model, analysis=analysis)


The ``info`` attribute shows the result in a readable format, showing that all 6 free parameters were fitted for.

.. code-block:: python

    print(result.info)

The output appears as follows:

.. code-block:: bash

    Bayesian Evidence                       144.86032973
    Maximum Log Likelihood                  181.14287034
    Maximum Log Posterior                   181.14287034

    model                                   Collection (N=6)
        gaussian                            Gaussian (N=3)
        exponential                         Exponential (N=3)

    Maximum Log Likelihood Model:

    gaussian
        centre                              50.223
        normalization                       26.108
        sigma                               9.710
    exponential
        centre                              50.057
        normalization                       39.948
        rate                                0.048


    Summary (3.0 sigma limits):

    gaussian
        centre                              50.27 (49.63, 50.88)
        normalization                       26.22 (21.37, 32.41)
        sigma                               9.75 (9.25, 10.27)
    exponential
        centre                              50.04 (49.60, 50.50)
        normalization                       40.06 (37.60, 42.38)
        rate                                0.05 (0.04, 0.05)


    Summary (1.0 sigma limits):

    gaussian
        centre                              50.27 (50.08, 50.49)
        normalization                       26.22 (24.33, 28.39)
        sigma                               9.75 (9.60, 9.90)
    exponential
        centre                              50.04 (49.90, 50.18)
        normalization                       40.06 (39.20, 40.88)
        rate                                0.05 (0.05, 0.05)

We can again use the max log likelihood instance to visualize the model data of the best fit model compared to the
data.

.. code-block:: python

    instance = result.max_log_likelihood_instance

    model_gaussian = instance.gaussian.model_data_1d_via_xvalues_from(
        xvalues=np.arange(data.shape[0])
    )
    model_exponential = instance.exponential.model_data_1d_via_xvalues_from(
        xvalues=np.arange(data.shape[0])
    )
    model_data = model_gaussian + model_exponential

    plt.errorbar(
        x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
    )
    plt.plot(range(data.shape[0]), model_data, color="r")
    plt.plot(range(data.shape[0]), model_gaussian, "--")
    plt.plot(range(data.shape[0]), model_exponential, "--")
    plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.show()
    plt.close()

The plot appears as follows:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/toy_model_fit.png
  :width: 600
  :alt: Alternative text

Cookbooks
----------

This overview shows the basics of model-fitting with **PyAutoFit**.

The API is designed to be intuitive and extensible, and you should have a good feeling for how you would define
and compose your own model, fit it to data with a chosen non-linear search, and use the results to interpret the
fit.

The following cookbooks give a concise API reference for using **PyAutoFit**, and you should use them as you define
your own model to get a fit going:

 - Model Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/model.html
 - Searches Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/analysis.html
 - Analysis Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/search.html
 - Results Cookbook: https://pyautofit.readthedocs.io/en/latest/cookbooks/result.html

There are additioal cookbooks which explain advanced PyAutoFit functionality
which you should look into after you have a good understanding of the basics.

The next overview describes how to set up a scientific workflow, where many other tasks required to perform detailed but
scalable model-fitting can be delegated to **PyAutoFit**. 
