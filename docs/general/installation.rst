.. _installation:

Installation
============

Dependencies
------------

This guide installs **PyAutoFit** with the following dependencies:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**Dynesty** https://github.com/joshspeagle/dynesty

**emcee** https://github.com/dfm/emcee

**PySwarms** https://github.com/ljvmiranda921/pyswarms

**astropy** https://www.astropy.org/

**corner.py** https://github.com/dfm/corner.py

**matplotlib** https://matplotlib.org/

**numpy** https://numpy.org/

**scipy** https://www.scipy.org/

Installation with pip
---------------------

The simplest way to install **PyAutoFit** is via pip:

.. code-block:: bash

    pip install autofit

Clone autofit workspace & set WORKSPACE environment model ('--depth 1' clones only the most recent branch on the
autofit_workspace, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Finally, run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py

Installation with conda
-----------------------

Installation via a conda environment circumvents compatibility issues when installing certain libraries.

First, install `conda <https://conda.io/miniconda.html>`_.

Create a conda environment:

.. code-block:: bash

    conda create -n autofit python=3.7 anaconda

Activate the conda environment:

.. code-block:: bash

    conda activate autofit

Install autofit:

.. code-block:: bash

    pip install autofit

Clone autofit workspace & set WORKSPACE environment model ('--depth 1' clones only the most recent branch on the
autofit_workspace, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

We will import files from the autofit_workspace as if it were a Python module. To do this in conda, we need to
create a .pth file in our conda enviroments site-packages folder. In your browser or on the command line find your
site packages folder:

.. code-block:: bash

   cd /home/usr/anaconda3/envs/autofit/lib/python3.7/site-packages/

Now create a .pth file via a text editor and put the path to your autofit_workspace in the file and save

NOTE: As shown below, the path in the .pth file points to the directory containing the 'autofit_workspace' folder
but does not contain the 'autofit_workspace' in PYTHONPATH itself!

.. code-block:: bash

   /path/on/your/computer/you/want/to/put/the

Finally, run the `welcome.py` script to get started!

.. code-block:: bash

   python3 welcome.py

Forking / Cloning
-----------------

If fork or clone the **PyAutoFit** github repository, note that **PyAutoFit** requires a valid autofit_workspace and
WORKSPACE environment to run (so it can find the necessary confgiuration files).

Therefore, if you fork or clone the **PyAutoFit** repository, you must also clone the
`autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_:

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autofit_workspace
   git clone https://github.com/Jammy2211/autofit_workspace --depth 1
   cd autofit_workspace

Once your fork of **PyAutoFit** is setup, I recommend you run the `welcome.py` script in the *autofit_workspace*
for an introduction to **PyAutoFit**.

.. code-block:: bash

   python3 welcome.py

Environment Variables
---------------------

**PyAutoFit** uses an environment variable called WORKSPACE to know where the 'autofit_workspace' folder is located.
This is used to locate config files and output results. It should automatically be detected and set in the `welcome.py`
script, but if something goes wrong you can set it manually using the command:

.. code-block:: bash

    export WORKSPACE=/path/on/your/computer/where/you/cloned/the/autofit_workspace

The autofit_workspace imports modules within the workspace to use them, meaning the path to the workspace must be
included in the PYTHONPATH. Your PYTHONPATH can be manual set using the command below.

NOTE: As shown below, the PYTHONPATH points to the directory containing the 'autofit_workspace' folder but does not
contain the 'autofit_workspace' in PYTHONPATH itself!

.. code-block:: bash

    export PYTHONPATH=/path/on/your/computer/you/want/to/put/the/.

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

Forking / Cloning
-----------------

Alternatively, you can fork or clone the **PyAutoFit** github repository. Note that **PyAutoFit** requires a valid
config to run. Therefore, if you fork or clone the **PyAutoFit** repository, you need the
`autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_ with the PYTHONPATH and WORKSPACE environment
variables set up as described on the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_ repository
or the installation instructions below.

Trouble Shooting
----------------

If you have issues with installation or using **PyAutoFit** in general, please raise an issue on the
`autofit_workspace issues page <https://github.com/Jammy2211/autofit_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).
