Chapter: Phase API
==================

In this chapter, we introduce you to the **PyAutoFit** phase API, which long-term software projects can adopt to interface
**PyAutoFit** directly with your software and provide a better management of many of model composition and fitting,
such as visualization, outputting results in a structured format and augmenting data before a fit.

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