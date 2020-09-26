Probibalistic Programming
=========================

Probabilistic programming languages (PPLs) have enabled contemporary statistical inference techniques to be applied
to a diverse range of problems across academia and industry. Packages such as
`PyMC3 <https://github.com/pymc-devs/pymc3>`_, `Pyro <https://github.com/pyro-ppl/pyro>`_ and
`STAN <https://github.com/stan-dev/stan>`_ offer general-purpose frameworks where users can specify a generative
model and fit it to data using a variety of non-linear fitting techniques. Each package is specialized to problems
of a certain nature, with many focused on problems like generalized linear modeling or determining the
distribution(s) from which the data was drawn. For these problems the model is typically composed of linear equations
which are easily expressed syntactically, such that the PPL API offers an expressive way to define the model and
extensions can be implemented in an intuitive and straightforward way.

Why PyAutoFit?
==============

**PyAutoFit** is a PPL whose core design is providing a direct interface with the model, data, fitting procedure and 
results, allowing it to provide more complete management of the model-fitting task than other PPLs and making it 
suited to longer term software projects. Model components are written as Python classes, allowing **PyAutoFit** to 
define the model and associated parameters in an expressive way that is tied to the modeling software's API. A 
model fit then only requires that a **PyAutoFit** ``Analysis`` class is writen, which combines the data, model and 
likelihood function and defines how the model-fit is performed using a non-linear search 
(e.g. `dynesty <https://github.com/joshspeagle/dynesty>`_, `emcee <https://github.com/dfm/emcee>`_
or `PySwarms <https://pyswarms.readthedocs.io/en/latest/>`_).

The ``Analysis`` class provides a model specific interface between **PyAutoFit** and the modeling software, allowing it 
to handle the 'heavy lifting' that comes with writing model-fitting software. This includes interfacing with the 
non-linear search, outputting results in a structured path format and model-specific visualization during and 
after the non-linear search. Results are output in a database structure with metadata that allows 
the ``Aggregator`` tool to load results post-analysis via a Python script or Jupyter notebook. This includes methods 
for summarizing the results of every fit, filtering results to inspect subsets of model fits and visualizing results. 
Results are loaded as ``Python`` generators, ensuring the ``Aggregator`` can be used to interpret large result datasets 
in a memory efficient way. **PyAutoFit** is therefore suited to 'big data' problems where independent fits to large 
homogeneous data-sets using an identical model-fitting procedure are performed. 

Model Abstract and Composition
==============================

For many modeling problems the model comprises abstract model components representing objects or processes in a
physical system. For example, our child project `PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_,  which uses
model components to reprsent the light and mass profiles of. For these problems the likelihood function is typically a
sequence of numerical processes (e.g. convolutions, Fourier transforms, linear algebra) and extensions to the model
often requires the addition of new model components in a way that is non-trivially included in the fitting process
and likelihood function. Existing PPLs have tools for these problems, for example `black-box' likelihood functions
in PyMC3. However, these solutions decouple model composition from the data and fitting procedure, making the model
less expressive, restricting model customization and reducing flexibility in how the model-fit is performed.

By writing model components as Python classes, the model and its associated parameters are defined in an expressive
way that is tied to the modeling software’s API. Model composition with **PyAutoFit** allows complex models to be built
from these individual components, abstracting the details of how they change model-fitting procedure from the user.
Models can be fully customized, allowing adjustment of individual parameter priors, the fixing or coupling of
parameters between model components and removing regions of parameter space via parameter assertions. Adding new model
components to a **PyAutoFit** project is straightforward, whereby adding a new Python class means it works within
the entire modeling framework. **PyAutoFit** is therefore ideal for problems where there is a desire to compose, fit and
compare many similar (but slightly different) models to a single dataset, with the **Aggregator** including tools to
facilitate this.

To see this in action, checkout the `overview section <https://pyautofit.readthedocs.io/en/latest/overview/model_fit.html>`_
of our readthedocs and the `HowToFit lectures <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_ for
guide on how to integrate **PyAutoFit** into your modeling software. More statistically minded readers may be interested
in **PyAutoFit**'s advanced statistical methods, such
as `transdimensional pipieline <https://pyautofit.readthedocs.io/en/latest/advanced/pipelines.html>`_.

.. toctree::
   :caption: General:
   :maxdepth: 1
   :hidden:

   general/installation
   general/workspace
   general/configs
   general/software
   general/citations
   general/future
   general/credits

.. toctree::
   :caption: Overview:
   :maxdepth: 1
   :hidden:

   overview/model_fit
   overview/model_complex
   overview/non_linear_search
   overview/result
   overview/aggregator
   overview/phase

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   howtofit/howtofit
   howtofit/chapter_1_introduction/index
   howtofit/chapter_2_results/index

.. toctree::
   :caption: API Reference:
   :maxdepth: 1
   :hidden:

   api/api

.. toctree::
   :caption: Advanced:
   :maxdepth: 1
   :hidden:

   advanced/pipelines