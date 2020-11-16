.. _model_fit:

Fitting a Model
---------------

To illustrate **PyAutoFit** we'll use the example modeling problem of fitting a 1D Gaussian profile to
noisy data.

The example ``data`` with errors (black) and the model-fit (red), are shown below:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/toy_model_fit.png
  :width: 600
  :alt: Alternative text

To begin, lets import ``autofit`` (and ``numpy``) using the convention below:

.. code-block:: bash

    import autofit as af
    import numpy as np

we need to define our 1D Gaussian profile as a **PyAutoFit** *model-component*. A *model component* is a component
of the model we fit to the ``data`` and it is has associated with it a set of *parameters* that can be varied for
during *model-fitting*.

*Model components* are defined using Python classes using the format below, where the class name is the *model component*
name and the constructor arguments are its *parameters*.

.. code-block:: bash

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

The code above defines a **PyAutoFit** *model component* called a ``Gaussian``. When used for *model-fitting* it has
three parameters: ``centre``, ``intensity`` and ``sigma``.

When we fit the model to ``data`` and compute a likelihood an instance of the class above is accessible, with specific
values of ``centre``, ``intensity`` and ``sigma`` chosen by the ``NonLinearSearch`` algorithm that fits the model to
the data.

This means that the class's functions are available to compute the likelihood, so lets add a ``profile_from_xvalues``
function that generates the 1D profile from the ``Gaussian``.

.. code-block:: bash

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

        def profile_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return np.multiply(
                np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
                np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
            )

Now we've defined our model, we need to tell **PyAutoFit** how to fit the model to data. This requires us to
define a **PyAutoFit** ``Analysis`` class:

.. code-block:: bash

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            # The 'instance' that comes into this method is an instance of the Gaussian
            # class, which the print statements below illustrates if you run the code!

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("Intensity = ", instance.intensity)
            print("Sigma = ", instance.sigma)

            # Get the range of x-values the data is defined on, to evaluate the model
            # of the Gaussian.

            xvalues = np.arange(self.data.shape[0])

            # Use these xvalues to create model_data of our Gaussian.

            model_data = instance.profile_from_xvalues(xvalues=xvalues)

            # Fit the model gaussian to the data, computing the residuals, chi-squareds
            # and returning the log likelihood value to the NonLinearSearch.

            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

Lets consider exactly what is happening in the ``Analysis`` class above.

- The ``data`` is passed into the constructor of the ``Analysis`` class. Above, only ``data`` and a ``noise_map`` are
  input, but the constructor can be easily extended to add other parts of the dataset.

- The ``log_likelihood_function`` receives an ``instance`` of the model, which in this example is an ``instance`` of the
  ``Gaussian`` class. This ``instance`` has values for its *parameters* (``centre``, ``intensity`` and ``sigma``) which are
  chosen by the ``NonLinearSearch`` used to fit the model, as discussed next.

- The ``log_likelihood_function`` returns a log likelihood value, which the ``NonLinearSearch`` uses evaluate the
  goodness-of-fit of a model to the data when sampling parameter space.

Next, we *compose* our model, set up our ``Analysis`` and fit the model to the ``data`` using a ``NonLinearSearch``:

.. code-block:: bash

    model = af.PriorModel(Gaussian)

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(name="example_search")

    result = emcee.fit(model=model, analysis=analysis)

Above, we use a ``PriorModel`` object to compose the model. This tells **PyAutoFit** that the ``Gaussian`` class is to
be used as a *model component* where its *parameters* are to be fitted for by the ``NonLinearSearch``. The model is
defined with 3 free parameters, thus the dimensionality of non-linear parameter space is 3.

We perform the fit using the ``NonLinearSearch`` algorithm `emcee <https://github.com/dfm/emcee>`_ (we cover
``NonLinearSearch``'s in more detail later).

By running the code above **PyAutoFit** performs the model-fit, outputting all results into structured paths on you
hard-disk. It also returns a ``Result`` object in Python, which includes lists containing the ``NonLinearSearch``'s
parameter samples, the maximum likelihood model, marginalized parameters estimates, errors are so on:

.. code-block:: bash

    print(result.samples.parameters)
    print(result.samples.max_log_likelihood_vector)
    print(result.samples.median_pdf_vector)
    print(result.samples.error_vector_at_sigma)

It can even return *instances* of the ``Gaussian`` class using the values of the model results:

.. code-block:: bash

    instance = result.max_log_likelihood_instance

    print("Maximum Likelihood Gaussian Instance:")
    print("Centre = ", instance.centre)
    print("Intensity = ", instance.intensity)
    print("Sigma = ", instance.sigma)

This can be used to straight forwardly plot the model fit to the data:

.. code-block:: bash

    instance = result.max_log_likelihood_instance

    model_data = instance.profile_from_xvalues(xvalues=np.arange(data.shape[0]))

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data)

This completes our basic introduction to the **PyAutoFit** API. Next, we'll cover how to *compose* and *fit*
models using multiple *model components* and *customize* the model parameterization.

If you'd like to perform the fit shown in this script, checkout the
`simple examples <https://github.com/Jammy2211/autofit_workspace/tree/master/examples/simple>`_ on the
``autofit_workspace``. We also detail how **PyAutoFit** works in the first 3 tutorials of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_.