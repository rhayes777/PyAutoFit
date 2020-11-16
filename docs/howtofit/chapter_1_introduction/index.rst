Chapter 1: Introduction
=======================

In chapter 1, we introduce you to the **PyAutoFit** API and describe how to set up your own software
project to use **PyAutoFit**.

A number of the notebooks require a ``NonLinearSearch`` to be performed, which can lead the auto-generation
of the **HowToFit** readthedocs pages to crash. For this reason, all cells which perform a ``NonLinearSearch``
or use its result are commented out. We advise if you want to read through the **HowToFit** lectures in full
that you download the ``autofit_workspace`` and run them from there (where these comments are removed).

The chapter contains the following tutorials:

`Tutorial 1: Model Mapping <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_1_model_mapping.html>`_
- The **PyAutoFit** backend that handles model composition.

`Tutorial 2: Model Fitting <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_2_model_fitting.html>`_
- Fitting a model with an input set of parameters to data.

`Tutorial 3: Non Linear Search <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_3_non_linear_search.html>`_
- Finding the model parameters that best-fit the data.

`Tutorial 4: Visualization <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_4_visualization.html>`_
- Outputting images during the model-fit and masking your data.

`Tutorial 5: Complex Models  <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_5_complex_models.html>`_
- Composing and fitting complex models.

`Tutorial 6: Results and Samples <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_6_results_and_samples.html>`_
- The results of a model-fit output by **PyAutoFit**.

`Tutorial 7: Fitting Multiple Datasets <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_7_fitting_multiple_datasets.html>`_
- Fitting a model to multiple similar datasets.

`Tutorial 8: Aggregator <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_8_aggregator.html>`_
- Loaing large libraries of results using in-built database tools.

`Tutorial 9: Filtering <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_9_filtering.html>`_
- Filtering the loaded results to find specific results of interest.

`Tutorial 10: Data and Models <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_1_introduction/tutorial_10_data_and_models.html>`_
- Replotting the model, data and fits of a set of results.

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   tutorial_1_model_mapping
   tutorial_2_model_fitting
   tutorial_3_non_linear_search
   tutorial_4_visualization
   tutorial_5_complex_models
   tutorial_6_results_and_samples
   tutorial_7_fitting_multiple_datasets
   tutorial_8_aggregator
   tutorial_9_filtering
   tutorial_10_data_and_models