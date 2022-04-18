.. _chapter_graphical_models:

Chapter: Graphical Models
=========================

In this chapter, we take you through how to compose and fit graphical models in **PyAutoFit**. Graphical models
simultaneously fit many datasets with a model that has 'local' parameters specific to each individual dataset
and 'global'parameters that fit for global trends across the whole dataset.

You can start the tutorials right now by going to `our binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD>`_
and navigating to the folder ``notebooks/howtofit/chapter_graphical_models``. They are also on the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_.

The chapter contains the following tutorials:

`Tutorial 1: Global Model <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_1_global_model.ipynb>`_
- An example of inferring global parameters from a dataset by fitting the model to each dataset one-by-one (and the downsides of doing this).

`Tutorial 2: Graphical Model <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_2_graphical_model.ipynb>`_
- Fitting the dataset with a graphical model that fits all datasets simultaneously to infer the global parameters (and the benefits of doing this).

`Tutorial 3: Hierarchical <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_3_hierarchical.ipynb>`_
- Fitting hieracrhical models using the graphical modeling framework.

`Tutorial 4: Expectation Propagation <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/release?filepath=notebooks/howtofit/chapter_graphical_models/tutorial_4_expectation_propagation.ipynb>`_
- Scaling graphical models up to fit extremely large datasets using Expectation Propagation (EP).

