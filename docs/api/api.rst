=============
API Reference
=============

.. currentmodule:: autofit

-------------------
Non-Linear Searches
-------------------

**Nested Samplers:**

.. autosummary::
   :toctree: generated/

   DynestyDynamic
   DynestyStatic
   MultiNest
   UltraNest

**MCMC:**

.. autosummary::
   :toctree: generated/

   Emcee
   Zeus

**Optimizers:**

.. autosummary::
   :toctree: generated/

   PySwarmsLocal
   PySwarmsGlobal

**GridSearch**:

.. autosummary::
   :toctree: generated/

   SearchGridSearch
   GridSearchResult

**Tools**:

.. autosummary::
   :toctree: generated/

   DirectoryPaths
   DatabasePaths
   Result
   InitializerBall
   InitializerPrior
   PriorPasser
   AutoCorrelationsSettings

--------
Plotters
--------

.. currentmodule:: autofit.plot
.. autosummary::
   :toctree: generated/

   DynestyPlotter
   UltraNestPlotter
   EmceePlotter
   ZeusPlotter
   PySwarmsPlotter


------
Models
------

.. currentmodule:: autofit

.. autosummary::
   :toctree: generated/

   PriorModel
   CollectionPriorModel


------
Priors
------

.. autosummary::
   :toctree: generated/

   UniformPrior
   GaussianPrior
   LogUniformPrior


-------
Samples
-------

.. autosummary::
   :toctree: generated/

   OptimizerSamples
   PDFSamples
   MCMCSamples
   NestSamples

----------
Aggregator
----------

.. autosummary::
   :toctree: generated/

   Aggregator

-------
Backend
-------

.. autosummary::
   :toctree: generated/

   ModelMapper
   ModelInstance
