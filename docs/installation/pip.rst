.. _pip:

Installation with pip
=====================

We strongly recommend that you install **PyAutoFit** in a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_, with the link attached
describing what a virtual enviroment is and how to create one.

**PyAutoFit** is via pip as follows:

.. code-block:: bash

    pip install autofit

If this raises no errors **PyAutoFit** is installed! If there is an error check out
the `troubleshooting section <https://pyautofit.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Next, clone the ``autofit workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autofit_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py