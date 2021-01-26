.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_
when you installed **PyAutoFit**. If you didn't, checkout the
`installation instructions <https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

Here, we give a brief tour of what is included in the workspace.

The following are provided as both `scripts` (e.g. .py files) and `notebooks` (Jupyter notebook .ipyn files).

Overview
--------

Example for fitting 1D data of noisy Gaussian profiles, including how to create an ``Analysis`` class, compose a model,
perform a model-fit, inspect results via the database tools. Two examples, illustrating ``simple`` and ``complex``
model-fits are provided.

HowToFit
--------

The **HowToFit** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoFit** model
fitting project and giving illustrations of different statistical methods and techniiques.

Checkout the
`tutorials section <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_ for a
full description of the lectures and online examples of every notebook.


Features
--------

Examples of **PyAutoFit**'s advanced model-fitting features, which are detailed in the 'Features' tab of the readthedocs.



The following folders are used by **PyAutoFit** for managing model-fitting.

Config
------

Here, you'll find the configuration files used by **PyAutoFit** which customize:

    - The default settings used by every ``NonLinearSearch``.
    - Example priors and notation configs which associate model-component with model-fitting.
    - The ``general.ini`` config which customizes other aspects of **PyAutoFit**.

Checkout the `configuration <https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the ``readthedocs`` for a complete description of every configuration file.

Dataset
-------

This folder stores the example dataset's used in the ``example`` package and will include the data fpr your
model-fitting problem.

Output
------

The folder where the model-fitting results of a ``NonLinearSearch`` are stored.