.. _api:

Non-linear Searches
-------------------

**PyAutoFit** currently supports three non-linear search algorithms, *emcee*, *Dynesty* and *PyMultiNest* (*PyMultiNest*
requires users to manually install it and is omitted from this example).

**PyAutoFit** extends *non-linear* search functionality to ensure that they always perform the following tasks, even if
the *non-linear search* package itself does not:

- Stores the results of the non-linear search to the hard-disk, writing the results to human-readable files.
- Allows the non-linear search to be resumed if a previous run was finished.
- Backs up results and associated metadata in .zip files, with the option to remove all other outputs for computing on
  HPCs where there may be file number quotas.
- Extends the functionality of the non-linear searches, for example providing auto-correlation analysis and stopping
  criteria for MCMC algorithms.

We've seen that we can call a *non-linear* search as follows:

.. code-block:: bash

   analysis = a.Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee()

   result = emcee.fit(model=model, analysis=analysis)

However, *emcee* has many settings associated with it (the number of walkers, the number of steps they take, etc.).
Above, we did not pass them to the *Emcee* constructor and they use the default values found in the *autofit_workspace*
configuration files 'autofit_workspace/config/non_linear/Emcee.ini', which can be viewed at this
`link <https://github.com/Jammy2211/autofit_workspace/blob/master/config/non_linear/Emcee.ini>`_.

Of course, we can instead manually specify all of the parameters:

.. code-block:: bash

   analysis = a.Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(
       nwalkers=50,
       nsteps=2000,
       initialize_method="ball",
       initialize_ball_lower_limit=0.49,
       initialize_ball_upper_limit=0.51,
       auto_correlation_check_for_convergence=True,
       auto_correlation_check_size=100,
       auto_correlation_required_length=50,
       auto_correlation_change_threshold=0.01,
   )

   result = emcee.fit(model=model, analysis=analysis)

A number of these parameters are not part of the *emcee* package, but additional functionality added by **PyAutoFit**:

- Initialization methods for the walkers are provided, incuding the strategy recommended at this
  `link <https://emcee.readthedocs.io/en/stable/user/faq/?highlight=ball#how-should-i-initialize-the-walkers>`_ where
  the walkers are initialized as a compact 'ball' in parameter space.

- Auto correlation lengths can be checked during sampling and used to determine whether the MCMC chains have converged,
  terminating *emcee* before all nwalkers have taken all nsteps, as discussed at this
  `link <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_.

The nested sampling algorithm *Dynesty* has its own config file for default settings, which are at this
`link <https://github.com/Jammy2211/autofit_workspace/blob/master/config/non_linear/Dynesty.ini>`_. Dynesty
parameters can be manually specified as follows:

.. code-block:: bash

   analysis = a.Analysis(data=data, noise_map=noise_map)

   dynesty = af.DynestyStatic(
       n_live_points=150,
       bound="multi",
       sample="auto",
       bootstrap=0,
       enlarge=-1,
       update_interval=-1.0,
       vol_dec=0.5,
       vol_check=2.0,
       walks=25,
       facc=0.5,
       slices=5,
       fmove=0.9,
       max_move=100,
       iterations_per_update=500,
   )

   result = dynesty.fit(model=model, analysis=analysis)

We can also customize the output folder and path structure where results are output. The output folder is set using
the **PyAutoFit** parent project **PyAutoConf** and the following command:

.. code-block:: bash

   from autoconf import conf

   conf.instance = conf.Config(output_path="path/to/output")

The path structure within this folder of a given *non-linear search* can be chosen using the *Paths* class when the
*non-linear search* is instantiated. For fits to many data-sets, this is important in ensuring results are clearly
labeled and the path where outputs occur do not clash.

The example code below would output the results to the path '/path/to/output/folder_0/folder_1/name/emcee':

.. code-block:: bash

   paths=af.Paths(folders=["folder_0", "folder_1"], name="name", non_linear_name="emcee")

Both *Emcee* and *Dynesty* support parallel analysis using the Python *multiprocessing* module. This distributes the
*non-linear search* analysis over multiple CPU's, speeding up the run-time roughly by the number of CPUs used. To
use this functionality in **PyAutoFit** you simply specifc the *number_of_cores* parameter (which is also found in the
default config files):

.. code-block:: bash

   analysis = a.Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(
       number_of_cores=4
   )

   result = emcee.fit(model=model, analysis=analysis)

.. code-block:: bash

   analysis = a.Analysis(data=data, noise_map=noise_map)

   dynesty = af.DynestyStatic(
       number_of_cores=4
   )

   result = dynesty.fit(model=model, analysis=analysis)

An immediate goal of **PyAutoFit** development is to add more *non-linear search* packages to the library. If you are
the developer of a package and would like it to get it implemented into **PyAutoFit** please contact us!