.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_
when you installed **PyAutoFit**. If you didn't, checkout the
`installation instructions <https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

Here, we give a brief tour of what is included in the workspace.

Config
------

Here, you'll find the configuration files used by **PyAutoFit** which customize:

    - The default settings used by every ``NonLinearSearch``.
    - Visualization, including the backend used by ``matplotlib``.
    - Example priors and notation configs which associate model-component with model-fitting.
    - The ``general.ini`` config which customizes other aspects of **PyAutoFit**.

Checkout the `configuration <https://pyautofit.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the ``readthedocs`` for a complete description of every configuration file.

Examples
--------

Example scripts using the 1D ``data`` fitting model are provided here, including scripts for creating an ``Analysis`` c
lass,performing a model-fit, inspecting results and using the ``Aggregator``. Two examples, illustrating ``simple`` and
``complex`` model-fits are provided.

HowToFit
--------

The **HowToFit** lecture series are a collection of Jupyter notebooks describing how to build a **PyAutoFit** model
fitting project and giving illustrations of different statistical methods and techniiques.

Checkout the
`tutorials section <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_ for a
full description of the lectures and online examples of every notebook.

Dataset
-------

The folder where ``dataset``'s for your model-fitting problem is stored. Example ``data`` for the 1D ``data`` fitting
problem are provided in the workspace.

Output
------

The folder where the model-fitting results of your model-fitting problem are stored.