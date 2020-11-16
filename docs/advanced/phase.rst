.. _phase:

Phase API
---------

The **PyAutoFit** API described in the overview section and `autofit_workspace/examples` package can be used to
perform simple model-fitting tasks in a matter of minutes. If this is all you intend to use **PyAutoFit** for then you
don't need to worry about the Phase API. However if you are developing a long term software project to perform
model-fitting, we advise you adopt **PyAutoFit**'s phase API.

The phase API requires you to write a ``phase`` package for your model-fitting software, which acts as the interface
between **PyAutoFit** and your source code. The ``phase`` package combines your project's *model-components* (e.g. the
``Gaussian`` and ``Exponential`` classes), the ``NonLinearSearch``'s and the ``Analysis`` class to provide a more
concise management of model-fitting in you software. This includes many laborious tasks that can often take a lot of
time to write code to perform correctly, such as outputting results to hard-disc, configuration files and visualization.

It also creates a simple interface for performing a *model-fits* for your users. Below we show how, if a project
adopts the Phase API, a model-fit can be set up in just two lines of Python:

 .. code-block:: bash

    # Set up a phase, which takes a name, the model and a `NonLinearSearch`.
    # The phase contains Analysis class 'behind the scenes', as well as taking
    # care of results output, visualization, etc.

    phase = af.Phase(
        search=af.Emcee(name="phase_example", nwalkers=50, nsteps=100),
        model=af.CollectionPriorModel(gaussian=Gaussian, exponential=Exponential),
    )

    # To perform a model fit, we simply run the phase with a dataset.

    result = phase.run(dataset=dataset)

Other than the concise interface, there are a number of benefits to adopting the ``phase`` API:

- The project can use our parent project **PyAutoConf** to handle configuration files, which control things like what
visualization is performed, how the model is fitted and the output path structure of results.

- ``Result``'s objects and ``Aggregator`` functionality can be extended to provide model-specific information and
  tools.

- Phases can be customized to augment the input ``data`` or alter the model-fitting behaviour in
  ways specific to your modeling problem. These automatically change the output path structure,
  making it straight forward to repeat fits to datasets with different settings or models.

Users who wish to write a ``phase`` package should checkout the relevent chapter in the **HowToFit** tutorials on the
autofit_workspace. This gives an example template project as well as a step-by-step guide on how to write a ``phase``
package. This includes recommendations on how to structure the project using a clean object-oriented design that
provides the best interface with **PyAutoFit**.