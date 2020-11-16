Chapter 2: Results
==================

In chapter 2, we introduce you to results that are output after a **PyAutoFit** analysis alongside the
``Aggregator`` tool which allows you to load, inspect and interpret the results of many model-fits to big
datasets.

A number of the notebooks require a ``NonLinearSearch`` to be performed, which can lead the auto-generation
of the **HowToFit** readthedocs pages to crash. For this reason, all cells which perform a ``NonLinearSearch``
or use its result are commented out. We advise if you want to read through the **HowToFit** lectures in full
that you download the ``autofit_workspace`` and run them from there (where these comments are removed).

The chapter contains the following tutorials:

`Tutorial 1: Source Code <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_phase_api/tutorial_1_source_code.html>`_
- How the source code of your software project should be structure to best use **PyAutoFit**.

`Tutorial 2: Visualization and Conigs <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_phase_api/tutorial_2_visualization_and_configs.html>`_
- How to set up your project to visualize results and use configuration files.

`Tutorial 3: Phase Customization <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_phase_api/tutorial_3_phase_customization.html>`_
- Augmenting the ``data`` and customizing the likelihood function of a model-fit.

`Tutorial 4: Aggregator <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_phase_api/tutorial_4_aggregator.html>`_
- Designing your project to be exploit **PyAutoFit**'s `Aggregator`.

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   tutorial_1_source_code
   tutorial_2_visualization_and_configs
   tutorial_3_phase_customization
   tutorial_4_aggregator