.. _installation:

Installation
============

Dependencies
------------

This guide installs **PyAutoFit** with the following dependencies:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**dynesty** https://github.com/joshspeagle/dynesty

**emcee** https://github.com/dfm/emcee

**PySwarms** https://github.com/ljvmiranda921/pyswarms

**astropy** https://www.astropy.org/

**corner.py** https://github.com/dfm/corner.py

**matplotlib** https://matplotlib.org/

**numpy** https://numpy.org/

**scipy** https://www.scipy.org/

Installation with pip
---------------------

The simplest way to install **PyAutoFit** is via `pip`:

.. code-block:: bash

    pip install autofit

Clone ``autofit_workspace`` (``--depth 1`` clones only the most recent branch on the autofit_workspace, reducing
the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Finally, run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py

Installation with conda
-----------------------

Installation via a ``conda`` environment circumvents compatibility issues when installing certain libraries.

First, install `conda <https://conda.io/miniconda.html>`_.

Create a ``conda`` environment:

.. code-block:: bash

    conda create -n autofit python=3.7 anaconda

Activate the ``conda`` environment:

.. code-block:: bash

    conda activate autofit

Install ``autofit``:

.. code-block:: bash

    pip install autofit

Clone the ``autofit_workspace`` (``--depth 1`` clones only the most recent branch on the autofit_workspace,
reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

Cloning / Forking
-----------------

You can clone (or fork) the **PyAutoFit** github repository and run it from the source code.

First, clone (or fork) the **PyAutoFit** GitHub repository:

.. code-block:: bash

    git clone https://github.com/Jammy2211/PyAutoFit

Next, install the **PyAutoFit** dependencies via pip:

.. code-block:: bash

   cd PyAutoFit
   pip install -r requirements.txt

Include the **PyAutoFit** source repository in your PYTHONPATH (noting that you must replace the text
``/path/to`` with the path to the **PyAutoFit** directory on your computer):

.. code-block:: bash

   export PYTHONPATH=$PYTHONPATH:/path/to/PyAutoFit

Finally, check the **PyAutoFit** unit tests run and pass (you may need to install pytest via
``pip install pytest``):

.. code-block:: bash

    cd /path/to/PyAutoFit
   python3 -m pytest

Current Working Directory
-------------------------

**PyAutoFit** scripts assume that the ``autofit_workspace`` directory is the Python working directory. This means 
that, when you run an example script, you should run it from the ``autofit_workspace`` as follows:

.. code-block:: bash

    cd path/to/autofit_workspace (if you are not already in the autofit_workspace).
    python3 examples/simple/fit.py

The reasons for this are so that **PyAutoFit** can:
 
 - Load configuration settings from config files in the ``autofit_workspace/config`` folder.
 - Load example data from the ``autofit_workspace/dataset`` folder.
 - Output the results of models fits to your hard-disk to the ``autofit/output`` folder. 
 - Import modules from the ``autofit_workspace``, for example ``from autofit_workspace.examples.simple import model as m``.

If you have any errors relating to importing modules, loading data or outputting results it is likely because you
are not running the script with the ``autofit_workspace`` as the working directory!

Matplotlib Backend
------------------

Matplotlib uses the default backend on your computer, as set in the config file:

.. code-block:: bash

    autofit_workspace/config/visualize/general.ini

If unchanged, the backend is set to 'default', meaning it will use the backend automatically set up for Python on
your system.

.. code-block:: bash

    [general]
    backend = default

There have been reports that using the default backend causes crashes when running the test script below (either the
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: bash

    [general]
    backend = TKAgg

Trouble Shooting
----------------

If you have issues with installation or using **PyAutoFit** in general, please raise an issue on the
`autofit_workspace issues page <https://github.com/Jammy2211/autofit_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).
