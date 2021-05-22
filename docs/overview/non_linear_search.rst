.. _non_linear_search:

Non-linear Searches
-------------------

**PyAutoFit** currently supports three types of non-linear search algorithms:

- **Optimizers**: ``PySwarms``.
- **MCMC**: ``emcee`` and ``Zeus``.
- **Nested Samplers**: ``dynesty``, ``UltraNest`` (optional) and ``PyMultiNest`` (optional).

**PyAutoFit** extends the functionality of each non-linear search to ensure that they always perform the
following tasks, even if the original package does not:

- Stores the results of the non-linear search to the hard-disk, writing the results to human-readable files.
- Allows the non-linear search to be resumed if a previous run was finished.
- Can write results and associated metadata to an sqlite database for querying and inspection post model-fit.
- Extends the functionality of the non-linear search's, for example providing auto-correlation analysis and
  stopping criteria for MCMC algorithms.

We've seen that we can call a non-linear search as follows:

.. code-block:: bash

   analysis = Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(name="example_mcmc")

   result = emcee.fit(model=model, analysis=analysis)

However, ``Emcee`` has many settings associated with it (the number of walkers, the number of steps they take,
etc.). Above, we did not pass them to the ``Emcee`` constructor and they use the default values found in the
``autofit_workspace`` configuration files ``autofit_workspace/config/non_linear/mcmc/Emcee.ini``, which can be
viewed at this `link <https://github.com/Jammy2211/autofit_workspace/blob/master/config/non_linear/mcmc/Emcee.ini>`_.

Of course, we can manually specify all of the parameters instead:

.. code-block:: bash

   analysis = Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(
       name="example_mcmc",
       nwalkers=50,
       nsteps=2000,
       initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
       auto_correlations_settings=af.AutoCorrelationsSettings(
           check_for_convergence=True,
           check_size=100,
           required_length=50,
           change_threshold=0.01,
       ),
   )

   result = emcee.fit(model=model, analysis=analysis)

A number of these parameters are not part of the ``emcee`` package, but additional functionality added by
**PyAutoFit**:

- Initialization methods for the walkers are provided, including the strategy recommended at
this `page <https://emcee.readthedocs.io/en/stable/user/faq/?highlight=ball#how-should-i-initialize-the-walkers>`_ where
the walkers are initialized as a compact 'ball' in parameter space.

- Auto correlation lengths can be checked during sampling and used to determine whether the MCMC chains have
converged, terminating ``emcee`` before all ``nwalkers`` have taken all ``nsteps``, as discussed at
this `link <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_.

The nested sampling algorithm ``dynesty`` has its own config file for default settings, which are at
this `link <https://github.com/Jammy2211/autofit_workspace/blob/master/config/non_linear/nest/Dynesty.ini>`_.
``DynestyStatic`` parameters can be manually specified as follows:

.. code-block:: bash

   analysis = Analysis(data=data, noise_map=noise_map)

   dynesty = af.DynestyStatic(
       name="example_nest",
       nlive=150,
       bound="multi",
       sample="auto",
       bootstrap=None,
       enlarge=None,
       update_interval=None,
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

We can also customize the output folder and path structure where results are output. The output folder is set
using the **PyAutoFit** parent project **PyAutoConf** and the following command:

.. code-block:: bash

   from autoconf import conf

   conf.instance.push(new_path="path/to/config", output_path="path/to/output")

The path structure within this folder of a given non-linear search is set using the ``path_prefix``.

Results are output to a folder which is a collection of random characters, which is the 'unique_identifier' of
the model-fit. This identifier is generated based on the model fitted and search used, such that an identical
combination of model and search generates the same identifier.

This ensures that rerunning an identical fit will use the existing results to resume the model-fit. In contrast, if
you change the model or search, a new unique identifier will be generated, ensuring that the model-fit results are
output into a separate folder.

The example code below would output the results to the
path ``/path/to/output/folder_0/folder_1/unique_tag/example_mcmc/sihfiuy838h``:

.. code-block:: bash

   emcee = af.Emcee(
       path_prefix="folder_0/folder_1/",
       name="example_mcmc"
   )

For model-fits to multiple datasets, checkout the `database feature <https://pyautofit.readthedocs.io/en/latest/features/database.html>`_
which gives more options for customizing how the unique identifier is generated.

Most searches support parallel analysis using the Python ``multiprocessing`` module. This distributes the
non-linear search analysis over multiple CPU's, speeding up the run-time roughly by the number of CPUs used.

The in-built parallelization of Libraries such as ``emcee`` and ``dynesty`` can be slow, because the default behaviour
is for them to pass the full likelihood function to every CPU. If this function includes a large dataset that is being
fitted, this can lead to long communication overheads and slow performance.

**PyAutoFit** implements *sneaky parallelization*, whereby the data is passed to every CPU before the model-fit. This
requires no extra user input and is performed by default. To perform a parallel search, you simply specify
the ``number_of_cores`` parameter (which is also found in the default config files):

.. code-block:: bash

   analysis = Analysis(data=data, noise_map=noise_map)

   emcee = af.Emcee(number_of_cores=4)

   result = emcee.fit(model=model, analysis=analysis)

We are always looking to add more non-linear searches to **PyAutoFit**. If you are the developer of a package check out
our `contributions section <https://github.com/rhayes777/PyAutoFit/blob/master/CONTRIBUTING.md>`_ and please
contact us!