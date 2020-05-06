.. _installation:

Installation
============

We strive to make **PyAutoFit** as easy to install as possible. However, one of our dependencies, **PyMultiNest**,
has proven difficult to install on many machines. We therefore recommend users to install **PyAutoFit** via conda
following the instructions below, which handles the installation of **PyMultiNest**.

Users who wish to install PyAutoFit via pip or fork the respository will need to install **PyMultiNest** manually.

Dependencies
------------

**PyAutoFit** dependencies can be found in the requirements.txt file and they are automatically installed with
**PyAutoFit**. The libraries are:

**PyAutoConf** https://github.com/rhayes777/PyAutoConf

**PyMultiNest** http://johannesbuchner.github.io/pymultinest-tutorial/install.html

**Dynesty** https://github.com/joshspeagle/dynesty

**emcee** https://github.com/dfm/emcee

**astropy** https://www.astropy.org/

**GetDist** https://getdist.readthedocs.io/en/latest/

**matplotlib** https://matplotlib.org/

**numpy** https://numpy.org/

**scipy** https://www.scipy.org/

Installation with conda
-----------------------

We recommend installation using a conda environment as this circumvents a number of compatibility issues when
installing **PyMultiNest**.

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


Installation with pip
---------------------

Installation is also available via pip, however there are reported issues with
installing **PyMultiNest** that can make installation difficult, see
`here <https://github.com/rhayes777/PyAutoFit/blob/master/INSTALL.notes>`_

If **PyMultiNest** has installed correctly you may install **PyAutoFit** via pip as follows.

.. code-block:: bash

    pip install autofit

Clone autofit workspace & set WORKSPACE enviroment model:

.. code-block:: bash

    cd /path/where/you/want/autofit_workspace
    git clone https://github.com/Jammy2211/autofit_workspace
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


You can test everything is working by running the example pipeline runner in the autofit_workspace

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