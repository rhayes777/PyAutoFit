.. _api:

Phase API
---------

Following the previous API examples, you are now ready to adopt **PyAutoFit** for your model-fitting problem. We
recommend that you have a go writing your own *model component* and *analysis* classes and perform a *non-linear
search*, to familiarize yourself with **PyAutoFit**.

To help, example codes for the 1D Gaussian line profile example and multi-component line profile example are available
on the autofit_workspace. These include scripts containing model components, *Analysis* classes, *model-fits*, result
inspection and aggregator use. They provide a simple template that can be adopted for your *model-fitting* software.

Advanced uses may wish to adopt the **PyAutoFit** *phase* API, which requires them to write a *phase* package for their
modeling software. The phase API provides a simple and concise interface for performing a *model-fit*:

 .. code-block:: bash

    # Set up a phase, which takes a name, the model and a non-linear search.
    # The phase contains Analysis class 'behind the scenes', as well as taking
    # care of results output, visualization, etc.

    phase = af.Phase(phase_name="phase_example", model=Gaussian, non_linear_class=af.Emcee)

    # To perform a model fit, we simply run the phase with a dataset.

    result = phase.run(dataset=dataset)

Other than the concise interface, there are a number of benefits to adopting the *phase* API:

- As eluded to in previous examples, *Results* objects and *Aggregator* functionality can be extended to provide
  model-specific information and tools.

- Phases can be customized to augment the input data or alter the model-fitting behaviour in standardized ways which
  automatically change the output path structure, making it straight forward to repeat fits to datasets with different
  settings or models.

Adopting the *phase* API is required to use some of **PyAutoFit**'s advanced *model-fitting* methods, including
*transdimensional model-fitting pipelines*.

Users who wish to write a *phase* package should checkout the **HowToFit** tutorials on the autofit_workspace. Chapter
1 provides a detailed introduction to **PyAutoFit**, giving a step-by-step guide on how to write a *phase* package.

Furthermore, these tutorials provide guidance on how to structure your entire software project in a clean
object-oriented fashion which maximizes the interface to **PyAutoFit**. The includes writing visualization methods that
can be easily called when using the *aggregator*, making the addition of new model components to the software straight
forward and packaging the data in a structure that requires the least user input for loading and fitting.