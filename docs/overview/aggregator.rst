.. _api:

Aggregator
----------

In the previous example, we discussed the *Results* object, which contains information on the *non-linear search*
samples, the best-fit model and parameter estimates and errors. If you are fitting a model to only one dataset, this
object suffices, but what if you are fitting the model to many datasets? How do you analyse, interpret and combine the
results?

Lets extend our example of fitting a 1D Gaussian line profile and pretend we've fitted 100 independent datasets
containing 1D Gaussians, such that the results of every *non-linear search* are in an ordered path structure on our
hard-disk. We can use the *Aggregator* to load the results of all 100 *non-linear searches*:

.. code-block:: bash

    output_path = "/path/to/gaussian_x100_fits"
    agg = af.Aggregator(directory=str(output_path))

We can now use the *aggregator* to load the *Samples* object of all 100 *model-fits*. This object (and all objects
returned by the *aggregator*) are returned as a generator, as opposed to a list, dictionary or other data-structure.
This is because generators do not store large arrays or classes in memory until they are used, ensuring that when we
are manipulating large sets of results we do not run out of memory!

.. code-block:: bash

    samples = agg.values("samples")

    parameters = [samps.parameters for samps in agg.values("samples")]
    log_likelihoods = [samps.log_likelihoods for samps in agg.values("samples")]
    instances = [samps.max_log_likelihood_instance for samps in agg.values("samples")]

The results above are returned as lists containing entries for every *model-fit*, in this case 100 fits:

.. code-block:: bash

    print("Instance Of Fit to First Dataset \n")
    print("centre = ", instances[0].centre)
    print("intensity = ", instances[0].intensity)
    print("sigma = ", instances[0].sigma)

    print("Instance Of Fit to Last Dataset \n")
    print("centre = ", instances[99].centre)
    print("intensity = ", instances[99].intensity)
    print("sigma = ", instances[99].sigma)

The *Aggregator* can be used in Python scripts, however we recommend users adopt Jupyter Notebooks when using the
*aggregator*. Notebooks allow results to be inspected and visualized with immediate feedback, such that one can more
readily interpret the results.

The aggregator contains tools for filtering the results, such the results of subsets of *model-fits* can be loaded.
The simplest way to do this is to simply require that the path the results are stored in contain a certain string
(or strings).

For example, we could require that the path contains the string 'gaussian_10', meaning we would only load the
results of the *model-fit* to the 10th Gaussian in our dataset:

.. code-block:: bash

    agg_filter = agg.filter(
        agg.directory.contains("gaussian_10")
    )

If users adopt the **PyAutoFit** *phase* API, they can use the in-built phase naming structure to filter results based
on phase name:

.. code-block:: bash

    phase_name = "phase__fit_1d_gaussian"
    agg_filter = agg.filter(agg.phase == phase_name)

The *phase* API also allows the the user to customize the aggregator to load model-specific results
including the dataset, masks and auxiliary information about the data:

.. code-block:: bash

    dataset = agg.values("dataset")
    mask = agg.values("mask")
    info = agg.values("info")

Lets pretend you fitted a dataset independently 3 times, and wish to combine these fits into one samples object such
that methods that return parameter estimates or errors use the combined fit. This can be done by simply adding the
*Samples* objects together:

.. code-block:: bash

    samples = list(agg.values("samples"))

    samples = samples[0] + samples[1] + samples[2]

    samples.most_probable_instance

If a subset of *model-fits* are incomplete or still running, the user can tell the *aggregator* to load only the
results of completed fits:

.. code-block:: bash

    output_path = "/path/to/gaussian_x100_fits"
    agg = af.Aggregator(directory=str(output_path), completed_only=True)

In the non-linear search API example, we discussed how *model-fit* results can be stored as .zip files to reduce the
number of files used, which may be important for HPC use which often have file use limits. The downside of this is it
means all results are contained in .zip which the user must unzip to access.

Fortunately, if you point the *Aggregator* to the path where the results are stored as .zip files, it will automatically
unzip all results making them easily accessible on the hard disk. This simply requires to run the command we showed
above:

.. code-block:: bash

    output_path = "/path/to/gaussian_x100_fits"
    agg = af.Aggregator(directory=str(output_path))