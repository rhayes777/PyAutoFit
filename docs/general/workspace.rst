.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_
when you installed **PyAutoFit**. If you didn't, checkout the
`installation instructions <https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

New users should begin by checking out the following parts of the workspace.

HowToFit
--------

The **HowToFit** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoFit** model
fitting project and giving illustrations of different statistical methods and techniiques.

Checkout the
`tutorials section <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_ for a
full description of the lectures and online examples of every notebook.

Scripts / Notebooks
-------------------

There are numerous example describing how perform model-fitting with **PyAutoFit** and providing an overview of its
advanced model-fitting features. All examples are provided as Python scripts and Jupyter notebooks.

A full description of the scripts available is given on
the `autofit workspace GitHub page <https://github.com/Jammy2211/autofit_workspace>`_.

Config
------

Here, you'll find the configuration files used by **PyAutoFit** which customize:

    - The default settings used by every non-linear search.
    - Example priors and notation configs which associate model-component with model-fitting.
    - The ``general.ini`` config which customizes other aspects of **PyAutoFit**.

Checkout the `configuration <https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the ``readthedocs`` for a complete description of every configuration file.

Dataset
-------

This folder stores the example dataset's used in examples in the workspace.

Output
------

The folder where the model-fitting results of a non-linear search are stored.