.. _scientific_workflow:

Scientific Workflow
===================

NOTE: A complete description of the scientific workflow, including Python code extracts and visuals, is not complete
yet. It will be written in the near future. This script gives a concise overview of how **PyAutoFit** can be used to
perform a scientific workflow.

The functionality necessary to develop an effective scientific workflow is described in the cookbooks, which are
available on readthedocs and in the `cookbooks` package of the workspace.


A scientific workflow is the series of tasks that someone performs to undertake scientific study though model fitting.
This includes fitting model a dataset, interpret the results and gaining insight and intuition for what works for their
problem.

Initially, the study will be performed on a small number of datasets (e.g. ~10s of datasets), as the user develops
their model and gains insight into what works well. This is a manual process of trial and error, which often involves
fitting many different models to the datasets and inspecting the result to gain insight on what models are good.
Their scientific workflow must be flexible enough to allow them to quickly fit many different models to their data,
and output the results in a format that is quick and easy to inspect and interpret.

Eventually, one may then scale up to a large number of datasets (e.g. ~1000s of datasets). Manual inspection of
individual results becomes infeasible, and the scientific workflow requires a more automated apporach to model fitting
and interpretation. This may also see the analysis move to a high performance computing, meaning that result output
must be suitable for this environment.

**PyAutoFit** enables the development of effective scientific workflows for both small and large datasets, thanks
to the following features:

- **On The Fly Feedback**: The results of a model-fit are shown to the user on-the-fly in Jupiter notebooks, providing quick feedback allowing them adapt their scientific workflow accordingly.

- **Hard-disk Output**: All results of an analysis can output to hard-disk with high levels of customization, ensuring that quick inspection of results ~100s of datasets is feasible.

- **Visualization**: Results output includes model specific visualization, allowing the user to produce custom plots that further streamline result inspection.

- **Model Composition**: The API for model composition is extensible, meaning a user can easily experiment with different model assumptions and priors, as well as comparing many different models.

- **Searches**: Support for a wide variety of non-linear searches (nested sampling, MCMC etc.) means a user can find the fitting algorithm that is optimal for their model and dataset.

- **Multiple Datasets**: Dedicated support for simultaneously fitting multiple datasets simultaneously means a user can easily combine different datasets in their analysis in order to fit more complex models.

- **Database**: The results of a model-fit can be output to a relational sqlite3 database, meaning that once you are ready to scale up to large datasets you can easily do so.