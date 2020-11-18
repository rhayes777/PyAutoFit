Configs
=======

The ``autofit_workspace`` includes configuration files that customize the behaviour of the ``NonLinearSearch``'s,
visualization and other aspects of **PyAutoFit**. Here, we describe how to configure **PyAutoFit** to use the configs
and describe every configuration file complete with input parameters.

Setup
-----

By default, **PyAutoFit** looks for the config files in a ``config`` folder in the current working directory, which is
why we run autofit scripts from the ``autofit_workspace`` directory.

The configuration path can also be set manually in a script using **PyAutoConf** and the following command (the path
to the ``output`` folder where the results of a ``NonLinearSearch`` are stored is also set below):

.. code-block:: bash

    from autoconf import conf

    conf.instance.push(
        config_path="path/to/config", output_path=f"path/to/output"
    )

general.ini
-----------

This config file is found at ``autofit_workspace/config/general.ini`` and contains the following sections and variables:

[output]
    log_file -> str
        The file name the logged output is written to (in the ``NonLinearSearch`` output folder).
    log_level -> str
        The level of logging.
    model_results_decimal_places -> int
        The number of decimal places the estimated values and errors of all parameters in the ``model.results`` file are
        output to.
    remove_files -> bool
        If `True`, all output files of a ``NonLinearSearch`` (e.g. samples, samples_backup, model.results, images, etc.)
        are deleted once the model-fit has completed.

        A .zip file of all output is always created before files are removed, thus results are not lost with this
        option turned on. If **PyAutoFit** does not find the output files of a model-fit (because they were removed) but
        does find this .zip file, it will unzip the contents and continue the analysis as if the files were
        there all along.

        This feature was implemented because super-computers often have a limit on the number of files allowed per
        user and the large number of files output by **PyAutoFit** can exceed this limit. By removing files the
        number of files is restricted only to the .zip files.
    grid_results_interval -> int
        For a ``GridSearch`` this interval sets after how many samples on the grid output is
        performed for. A ``grid_results_interval`` of -1 turns off output.

non_linear
----------

These config files are found at ``autofit_workspace/config/non_linear`` and they contain the default settings used by
every ``NonLinearSearch``. The ``[search]``, ``[settings]`` and ``[initialize]`` sections of the non-linear configs
contains settings specific to certain ``NonLinearSearch``'s, and the documentation for these variables should be found
by inspecting the`API Documentation <https://pyautofit.readthedocs.io/en/latest/api/api.html>`_ of the relevent
``NonLinearSearch`` object.

The following config sections and variables are generic across all ``NonLinearSearch`` configs (e.g.
``config/non_linear/nest/DynestyStatic.ini``, ``config/non_linear/mcmc/Emcee.ini``, etc.):

[updates]
   iterations_per_update -> int
        The number of iterations of the ``NonLinearSearch`` performed between every 'update', where an update performs
        visualization of the maximum log likelihood model, backing-up of the samples, output of the ``model.results``
        file and logging.
   visualize_every_update -> int
        For every ``visualize_every_update`` updates visualization is performed and output to the hard-disk during the
        non-linear using the maximum log likelihood model. A ``visualization_interval`` of -1 turns off on-the-fly
        visualization.
   backup_every_update -> int
        For every ``backup_every_update`` the results of the ``NonLinearSearch`` in the samples foler and backed up into the
        samples_backup folder. A ``backup_every_update`` of -1 turns off backups during the ``NonLinearSearch`` (it is still
        performed when the ``NonLinearSearch`` terminates).
   model_results_every_update -> int
        For every ``model_results_every_update`` the model.results file is updated with the maximum log likelihood model
        and parameter estimates with errors at 1 an 3 sigma confidence. A ``model_results_every_update`` of -1 turns off
        the model.results file being updated during the model-fit (it is still performed when the ``NonLinearSearch``
        terminates).
   log_every_update -> int
        For every ``log_every_update`` the log file is updated with the output of the Python interpreter. A
        ``log_every_update`` of -1 turns off logging during the model-fit.

[printing]
    silence -> bool
        If `True`, the default print output of the ``NonLinearSearch`` is silenced and not printed by the Python
        interpreter.

[parallel]
    number_of_cores -> int
        For ``NonLinearSearch``'s that support parallel procesing via the Python ``multiprocesing`` module, the number of
        cores the parallel run uses. If ``number_of_cores=1``, the model-fit is performed in serial omitting the use
        of the ``multiprocessing`` module.

The output path of every ``NonLinearSearch`` is also 'tagged' using strings based on the ``[search]`` setting of the
``NonLinearSearch``:

[tag]
    name -> str
        The name of the ``NonLinearSearch`` used to start the tag path of output results. For example for the
        search ``DynestyStatic`` the default name tag is 'dynesty_static'.

visualize
---------

These config files are found at ``autofit_workspace/config/visualize`` and they contain the default settings used by
visualization in **PyAutoFit**. The ``general.ini`` config contains the following sections and variables:

[general]
    backend -> str
        The ``matploblib backend`` used for visualization (see
        https://gist.github.com/CMCDragonkai/4e9464d9f32f5893d837f3de2c43daa4 for a description of backends).

        If you use an invalid backend for your computer, **PyAutoFit** may crash without an error or reset your machine.
        The following backends have worked for **PyAutoFit** users:

        TKAgg (default)

        Qt5Agg (works on new MACS)

        Qt4Agg

        WXAgg

        WX

        Agg (outputs to .fits / .png but doesn't'display figures during a run on your computer screen)

priors
------

These config files are found at ``autofit_workspace/config/priors`` and they contain the default priors and related
variables for every model-component in a project, using ``.json`` format files (as opposed to ``.ini`` for most config files).

The autofit_workspace`` contains example ``prior`` files for the 1D ``data`` fitting problem. An example entry of the
json configs for the ``sigma`` parameter of the ``Gaussian`` class is as follows:

.. code-block:: bash

    "Gaussian": {
        "sigma": {
            "type": "Uniform",
            "lower_limit": 0.0,
            "upper_limit": 30.0,
            "width_modifier": {
                "type": "Absolute",
                "value": 0.2
            },
            "gaussian_limits": {
                "lower": 0.0,
                "upper": "inf"
            }
        },

The sections of this example config set the following:

json config
    type -> Prior
        The default prior given to this parameter which is used by the ``NonLinearSearch``. In the example above, a
        ``UniformPrior`` is used with ``lower_limit`` of 0.0 and ``upper_limit`` of 30.0. A ``GaussianPrior`` could be used by
        putting "``Gaussian``" in the "``type``" box, with "``mean``" and "``sigma``" used to set the default values. Any prior can be
        set in an analogous fashion (see the example configs).
    width_modifier
        When the results of a phase are linked to a subsequent phase to set up the priors of its ``NonLinearSearch``,
        this entry describes how the ``Prior`` is passed. For a full description of prior passing, checkout the examples
        in ``autofit_workspace/examples/complex/linking``.
    gaussian_limits
        When the results of a phase are linked to a subsequent phase, they are passed using a ``GaussianPrior``. The
        ``gaussian_limits`` set the physical lower and upper limits of this ``GaussianPrior``, such that parameter samples
        can not go beyond these limits.

notation
--------

The notation configs define the labels of every model-component parameter and its derived quantities, which are
used when visualizing results (for example labeling the axis of the PDF triangle plots output by a ``NonLinearSearch``).
Two examples using the 1D ``data`` fitting example for the config file **label.ini** are:

[label]
    centre -> str
        The label given to that parameter for ``NonLinearSearch`` plots using that parameter, e.g. the PDF plots. For
        example, if centre=x, the plot axis will be labeled 'x'.

[subscript]
    Gaussian -> str
        The subscript used on certain plots that show the results of different model-components. For example, if
        Gaussian=g, plots where the Gaussian are plotted will have a subscript g.

The **label_format.ini** config file specifies the format certain parameters are output as in output files like the
*model.results* file.