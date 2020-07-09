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

**GetDist** https://getdist.readthedocs.io/en/latest/

**matplotlib** https://matplotlib.org/

**numpy** https://numpy.org/

**scipy** https://www.scipy.org/

Installation with pip
---------------------

The simplest way to install **PyAutoFit** is via pip:

.. code-block:: bash

    pip install autofit

Clone autofit workspace & set WORKSPACE environment model ('--depth 1' clones only the most recent branch on the
autofit_workspace, reducing the download size)::

.. code-block:: bash

    cd /path/where/you/want/autofit_workspace
    git clone https://github.com/Jammy2211/autofit_workspace --depth 1
    export WORKSPACE=/path/to/autofit_workspace/

Set PYTHONPATH to include the autofit_workspace directory:

.. code-block:: bash

    export PYTHONPATH=/path/to/autofit_workspace

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

You can test everything is working by running the following command in the autofit_workspace:

.. code-block:: bash

    python3 /path/to/autofit_workspace/examples/simple/fit.py

PyMultiNest
-----------

Installation via pip omits an optional dependency, the nested sampling algorithm
`PyMultiNest <http://johannesbuchner.github.io/pymultinest-tutorial/install.html>`_. If you require **PyMultiNest** you
either need too install **PyAutoFit** via conda following the instructions below or will need to install **MultiNest**
`at this link <http://johannesbuchner.github.io/pymultinest-tutorial/install.html>`_.

Installation with conda
-----------------------

First, install `conda <https://conda.io/miniconda.html>`_.

Create a conda environment:

.. code-block:: bash

    >> conda create -n autofit python=3.7 anaconda


Activate the conda environment:

.. code-block:: bash

    conda activate autofit


Install multinest:

.. code-block:: bash

    conda install -c conda-forge multinest


Install autofit:

.. code-block:: bash

    pip install autofit


Clone the autofit workspace & set WORKSPACE environment model:

.. code-block:: bash

    cd /path/where/you/want/autofit_workspace
    git clone https://github.com/Jammy2211/autofit_workspace
    export WORKSPACE=/path/to/autofit_workspace/


Set PYTHONPATH to include the autofit_workspace directory:

.. code-block:: bash

    export PYTHONPATH=/path/to/autofit_workspace/

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


You can test everything is working by running the example pipeline runner in the autofit_workspace

.. code-block:: bash

    python3 /path/to/autofit_workspace/runners/beginner/no_fit_light/fit_sie__source_inversion.py

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
