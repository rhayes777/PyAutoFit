Chapter 1: Introduction
=======================

In chapter 1, we'll introduce you to the **PyAutoFit** API and describe how to set up your own software project to use
**PyAutoFit**.

A number of the notebooks require a *non-linear search* to be performed, which can lead the auto-generation of the
**HowToFit** readthedocs pages to crash. For this reason, all cells which perform a *non-linear search* or use its
result are commented out. We advise if you want to read through the **HowToFit** lectures in full that you download
the autofit_workspace and run them from there (where these comments are removed).

The chapter contains the following tutorials:

`Tutorial 1: Model Mapping <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_1_model_mapping.html>`_
- The **PyAutoFit** backend that handles model composition.

`Tutorial 2: Model Fitting <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_2_model_fitting.html>`_
- Fitting a model with an input set of parameters to data.

`Tutorial 3: Non Linear Search <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_3_non_linear_search.html>`_
- Finding the model parameters that best-fit the data.

`Tutorial 4: Source Code <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_4_source_code.html>`_
- How the source code of your software project should be structure to best use **PyAutoFit**.

`Tutorial 5: Visualization & Masking <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_5_visualization_masking.html>`_
- Outputting images during the model-fit and masking your data.

`Tutorial 6: Complex Models  <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_6_complex_models.html>`_
- Composing and fitting complex models.

`Tutorial 7: Phase Customization <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_7_phase_customization.html>`_
- Augmenting the data and customizing the likelihood function of a model-fit.

`Tutorial 8a: Aggregator <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_8_aggregator_part_1.html>`_
- Loading large suites of model-fitting results output by **PyAutoFit**.

`Tutorial 8b: Aggregator <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_8_aggregator_part_2.html>`_
- Tools for managing large suites of model-fitting results.

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   tutorial_1_model_mapping
   tutorial_2_model_fitting
   tutorial_3_non_linear_search
   tutorial_4_source_code
   tutorial_5_visualization_masking
   tutorial_6_complex_models
   tutorial_7_phase_customization
   tutorial_8_aggregator_part_1
   tutorial_8_aggregator_part_2