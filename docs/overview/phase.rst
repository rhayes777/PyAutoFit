.. _phase:

Phase API
---------

Following the previous API examples, you are now ready to adopt **PyAutoFit** for your model-fitting problem. We
recommend that you checkout the ``autofit_workspace`` examples we linked to at the end of page, have a read
through the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_ and
have a go at fitting your own model using an ``Analysis`` class!

Experienced uses may wish to adopt the **PyAutoFit** ``phase`` API, which requires them to write a ``phase``
package for their Python modeling package. The ``phase`` API combine the *model-components* (e.g. things like
the ``Gaussian`` and ``Exponential`` classes), ``NonLinearSearch``'s and the ``Analysis`` to provides a simple
and concise interface for performing a *model-fit*:

 .. code-block:: bash

    # Set up a phase, which takes a name, the model and a `NonLinearSearch`.
    # The phase contains Analysis class 'behind the scenes', as well as taking
    # care of results output, visualization, etc.

    phase = af.Phase(
        phase_name="phase_example",
        model=af.CollectionPriorModel(gaussian=Gaussian, exponential=Exponential),
        search=af.Emcee(nwalkers=50, nsteps=100)
    )

    # To perform a model fit, we simply run the phase with a dataset.

    result = phase.run(dataset=dataset)

Other than the concise interface, there are a number of benefits to adopting the ``phase`` API:

- As eluded to in previous examples, ``Result``'s objects and ``Aggregator`` functionality can be
  extended to provide model-specific information and tools.

- Phases can be customized to augment the input ``data`` or alter the model-fitting behaviour in
  ways specific to your modeling problem. These automatically change the output path structure,
  making it straight forward to repeat fits to datasets with different settings or models.

The ``phase`` API is required to use some of **PyAutoFit**'s advanced *model-fitting* methods, including
`transdimensional pipelines <https://pyautofit.readthedocs.io/en/latest/advanced/pipelines.html>`_.

Users who wish to write a ``phase`` package should checkout the **HowToFit** tutorials on the autofit_workspace.
Chapter 1 provides a detailed introduction to **PyAutoFit**, giving a step-by-step guide on how to write a
``phase`` package. This includes recommendations on how to structure the project using a clean object-oriented
design that provides a clean interface to **PyAutoFit**.