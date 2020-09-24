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

`Tutorial 1: Results and Samples <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_2_results/tutorial_1_model_mapping.html>`_
- The **PyAutoFit** backend that handles model composition.

`Tutorial 2: Fitting Multiple Datasets <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_2_results/tutorial_2_model_fitting.html>`_
- Fitting a model with an input set of parameters to data.

`Tutorial 3: Aggregator <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_2_results/tutorial_3_non_linear_search.html>`_
- Finding the model parameters that best-fit the data.

`Tutorial 4: Filtering <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_2_results/tutorial_4_source_code.html>`_
- How the source code of your software project should be structure to best use **PyAutoFit**.

`Tutorial 5: Data and Models <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_2_results/tutorial_5_visualization_masking.html>`_
- Outputting images during the model-fit and masking your data.

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   tutorial_1_results_and_samples
   tutorial_2_fitting_multiple_datasets
   tutorial_3_aggregator
   tutorial_4_filtering
   tutorial_5_data_and_models