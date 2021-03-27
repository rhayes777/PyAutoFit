.. _howtofit:

HowToFit Lectures
=================

To learn how to use **PyAutoFit**, the best starting point is the **HowToFit** lecture series, which are found on
the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_ and at
our `binder <https://mybinder.org/v2/gh/Jammy2211/autofit_workspace/HEAD>`_.

The lectures are provided as *Jupyter notebooks* and currently consist of 3 chapters:

**Introduction**: How to perform model-fitting with **PyAutoFit** and analyse the results using the ``Aggregator``.

**Graphical Models**: How to compose and fit graphical models to large datasets (notebooks + feature are in development).

Config File Path
----------------

If, when running the first notebook, you get an error related to config files, this most likely means that
**PyAutoFit** is unable to find the config files in your autofit workspace. Checkout the
`configs section <https://pyautofit.readthedocs.io/en/latest/general/configs.html>`_ for a description of how to
fix this.

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