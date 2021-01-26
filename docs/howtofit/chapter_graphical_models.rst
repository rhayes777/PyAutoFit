.. _chapter_graphical_models:

Chapter: Graphical Models
=========================

NOTE: This is an in development feature and this chapter is incomplete.

In this chapter, we take you through how to compose and fit graphical models in **PyAutoFit**. Graphical models
can fit large datasets with a model that has 'local' parameters specific to each individual dataset and 'global'
parameters that fit for higher level parameters.

You can start the tutorials right now by going to `our binder <https://notebooks.gesis.org/binder/v2/gh/Jammy2211/autofit_workspace/7586a67b726dca612404cf5fab1d77d8738f3737>`_
and navigating to the folder `notebooks/howtofit/chapter_phase_api`. They are also on the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_.

The chapter contains the following tutorials:

**Tutorial 1: Global Model**
- An example of inferring global parameters from a dataset by fitting the model to each dataset one-by-one.

**Tutorial 2: Graphical Model**
- Fitting the dataset with a graphical model that fits all datasets simultaneously to infer the global parameters