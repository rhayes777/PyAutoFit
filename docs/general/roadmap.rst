.. _roadmap:

Road Map
========

**PyAutoFit** is in active development and the road-map of features currently planned in the short and long term are
listed and described below:

**Non-Linear Searches:**

We are always striving to add new non-linear searches to **PyAutoFit*. In the short term, we aim to provide a wrapper
to the many method available in the `scipy.optimize` library with support for outputting results to hard-disk.

If you would like to see a non-linear search implemented in **PyAutoFit** please `raise an issue on GitHub <https://github.com/rhayes777/PyAutoFit/issues>`_!

**SQLite Relational Database:**

**PyAutoFit**'s database tools currently write and load results via the output paths on the hard-disk. This is
inefficient and restricts the type of filtering that is applicable when loading results from the database.

We are developing new database tools that write results to a relational database using the Python library SQLite. This
will ensure efficient database querying for extremely large model-fitting databases and allow for advanced querying
when loading results.

**Graphical Models**

Graphical models allow one to compose complex models that fit for global trends in many model-fits to individual
datasets.

Graphical models will support an `expectation propagation framework <https://arxiv.org/abs/1412.4869>`_ that breaks
the model-fitting of extremely large datasets down into many smaller fits. The results of these fits will be passed
through the graph to refine inference of the local parameters tied to each dataset and the global parameters
describing the graphical model.

**Approximate Bayesian Computation**

Approximate Bayesian Computational (ABC) allows for one to infer parameter values for likelihood functions that are
intractable, by simulating many datasets and extracting from them a summary statistic that is compared to the
observed dataset.

ABC in **PyAutoFit** will be closely tied to the Database tools, ensuring that the simulation, fitting and extraction
of summary statistics can be efficiently scaled up to extremely large datasets.