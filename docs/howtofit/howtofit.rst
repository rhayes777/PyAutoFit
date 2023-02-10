.. _howtofit:

HowToFit Lectures
=================

To learn how to use **PyAutoFit**, the best starting point is the **HowToFit** lecture series, which are found on
the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_ and at
our `binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD>`_.

The lectures are provided as *Jupyter notebooks* and currently consist of 3 chapters:

- ``chapter_1_introduction``: Introduction lectures describing how to compose and fit a models in **PyAutoFit**.
- ``chapter_database``: How to output results to a .sqlite database format and load them for analysis after model-fitting is complete.
- ``chapter_graphical_models``: How to compose and fit graphical models which fit many datasets simultaneously.

Statistics and Theory
---------------------

**HowToFit** assumes minimal previous knowledge of statistics, model-fitting and Bayesian inference. However, it is beneficial
yourself a basic theoretical grounding as you go through the lectures. I heartily recommend you aim to read up on
the concepts introduced throughout the lectures once you understand how to use them in **PyAutoFit**.

Jupyter Notebooks
-----------------

The tutorials are supplied as *Jupyter notebooks*, which come with a ``.ipynb`` suffix. For those new to
Python, *Jupyter notebooks* are a different way to write, view and use Python code. Compared to the
traditional Python scripts, they allow:

- Small blocks of code to be viewed and run at a time
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the HowToFit lecture series, therefore I recommend you get
yourself a Jupyter notebook viewer (https://jupyter.org/) if you havent done so already.

If you *really* want to use Python scripts, all tutorials are supplied a ``.py`` python files in the ``scripts``
folder of each chapter.

For actual **PyAutoFit** use I recommend you use Python scripts. Therefore, as you go through the lecture
series you will notice that we will transition you to Python scripts.

Code Style and Formatting
-------------------------

You may notice the style and formatting of our Python code looks different to what you are used to. For
example, it is common for brackets to be placed on their own line at the end of function calls, the inputs
of a function or class may be listed over many separate lines and the code in general takes up a lot more
space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact, lots
of people do, which is why we use an auto-formatter to produce the code in a standardized format. If you're
interested in the style and would like to adapt it to your own code, check out the Python auto-code formatter
``black``.

https://github.com/python/black

How to Approach HowToFit
------------------------

The **HowToFit** lecture series current sits at 3 chapters, and each will take more than a couple of hours to go through
properly. You probably want to be begin modeling with **PyAutoFit** faster than that! Furthermore, the concepts in the
later chapters are pretty challenging, and familiarity with **PyAutoFit** and model fitting is desirable before you
tackle them.

Therefore, we recommend that you complete chapter 1 and then apply what you've learnt to a model fitting problem you are
interested in, building on the scripts found in the 'autofit_workspace/examples' folder. Once you're happy
with the results and confident with your use of **PyAutoFit**, you can then begin to cover the advanced functionality
covered in other chapters.

Overview of Chapter 1 (Beginner)
--------------------------------

**Model Fitting with PyAutoFit**

In chapter 1, we'll learn about model composition and fitting and **PyAutoFit**. At the end, you'll
be able to:

1) Compose a model in **PyAutoFit**.
2) Define a ``log_likelihood_function()`` via an ``Analysis`` class to fit that model to data.
3) The concept of a non-linear search and non-linear parameter space.
4) Fit a model to data using a non-linear search.
5) Compose and fit complex models using **PyAutoFit**'s advanced model composition API.
6) Analyse the results of a model-fit, including parameter estimates and errors.

Overview of Chapter Database
----------------------------

**Writing Large Suites of Results to an SQLite3 Database**

Here, we learn how to use **PyAutoFit**'s sqlite3 database feature. You'll be able too:

1) Write results to the database.
2) Load results from the database and analyse them (e.g. parameter estimates, errors).
3) Query the database to get subsets of results.
4) Interface the database and your model-fitting code to perform custom tasks.

Overview of Chapter Graphical Models
------------------------------------

**Fitting a Graphical Model to a Large Dataset**

Here, we learn how to compose and fit graphical models to extremely large datasets. You'll learn:

1) Why fitting a model to many datasets one-by-one is suboptimal.
2) How to fit a graphical to model all datasets simultaneously and why this improves the model results.
3) Scaling graphical model fits up to extremely large datasets via Expectation Propagation.
4) How to fit a hierachical model using the graphical modeling framework and Expectation Propagation.