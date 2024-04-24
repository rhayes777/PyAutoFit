.. _multiple_datasets:

Multiple Datasets
=================

This cookbook illustrates how to fit multiple datasets simultaneously, where each dataset is fitted by a different
``Analysis`` class.

The ``Analysis`` classes are summed together to give an overall log likelihood function that is the sum of the
individual log likelihood functions, which is sampled by the non-linear search.

If one has multiple observations of the same signal, it is often desirable to fit them simultaneously. This ensures
that better constraints are placed on the model, as the full amount of information in the datasets is used.

In some scenarios, the signal may vary across the datasets in a way that requires that the model is updated
accordingly. **PyAutoFit** provides tools to customize the model composition such that specific parameters of the model
vary across the datasets.

This cookbook illustrates using observations of 3 1D Gaussians, which have the same ``centre`` (which is the same
for the model fitted to each dataset) but different ``normalization`` and ``sigma`` values (which vary for the model
fitted to each dataset).

It is common for each individual dataset to only constrain specific aspects of a model. The high level of model
customization provided by **PyAutoFit** ensures that composing a model that is appropriate for fitting large and diverse
datasets is straight forward. This is because different ``Analysis`` classes can be written for each dataset and summed.

**Contents:**

- **Model-Fit**: Setup a model-fit to 3 datasets to illustrate multi-dataset fitting.
- **Analysis Summing**: Sum multiple ``Analysis`` classes to create a single ``Analysis`` class that fits all 3 datasets simultaneously, including summing their individual log likelihood functions.
- **Result List**: Use the output of fits to multiple datasets which are a list of ``Result`` objects.
- **Variable Model Across Datasets**: Fit a model where certain parameters vary across the datasets whereas others stay fixed.
- **Relational Model**: Fit models where certain parameters vary across the dataset as a user defined relation (e.g. ``y = mx + c``).
- **Different Analysis Classes**: Fit multiple datasets where each dataset is fitted by a different ``Analysis`` class, meaning that datasets with different formats can be fitted simultaneously.
- **Interpolation**: Fit multiple datasets with a model one-by-one and interpolation over a smoothly varying parameter (e.g. time) to infer the model between datasets.
- **Individual Sequential Searches**: Fit multiple datasets where each dataset is fitted one-by-one sequentially.
- **Hierarchical / Graphical Models**: Use hierarchical / graphical models to fit multiple datasets simultaneously, which fit for global trends in the model across the datasets.

Model Fit
---------

Load 3 1D Gaussian datasets from .json files in the directory ``autofit_workspace/dataset/``.

All three datasets contain an identical signal, therefore fitting the same model to all three datasets simultaneously
is appropriate.

Each dataset has a different noise realization, therefore fitting them simultaneously will offer improved constraints 
over individual fits.

.. code-block:: python

    dataset_size = 3

    data_list = []
    noise_map_list = []

    for dataset_index in range(dataset_size):
        dataset_path = path.join(
            "dataset", "example_1d", f"gaussian_x1_identical_{dataset_index}"
        )

        data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
        data_list.append(data)

        noise_map = af.util.numpy_array_from_json(
            file_path=path.join(dataset_path, "noise_map.json")
        )
        noise_map_list.append(noise_map)

Plot all 3 datasets, including their error bars. 

.. code-block:: python

    for data, noise_map in zip(data_list, noise_map_list):
        xvalues = range(data.shape[0])

        plt.errorbar(
            x=xvalues,
            y=data,
            yerr=noise_map,
            color="k",
            ecolor="k",
            linestyle=" ",
            elinewidth=1,
            capsize=2,
        )
        plt.show()
        plt.close()

Here is what the plots look like:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_data_0.png
  :width: 300
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_data_1.png
  :width: 300
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_data_2.png
  :width: 300
  :alt: Alternative text

Create our model corresponding to a single 1D Gaussian that is fitted to all 3 datasets simultaneously.

.. code-block:: python

    model = af.Model(af.ex.Gaussian)

    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.GaussianPrior(
        mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
    )

Analysis Summing
----------------

Set up three instances of the ``Analysis`` class which fit 1D Gaussian.

We set up an ``Analysis`` for each dataset one-by-one, using a for loop.

.. code-block:: python

    analysis_list = []

    for data, noise_map in zip(data_list, noise_map_list):
        analysis = af.ex.Analysis(data=data, noise_map=noise_map)
        analysis_list.append(analysis)

We now sum together every analysis in the list, to produce an overall analysis class which we fit with the non-linear
search.

By summing analysis objects the following happen:

- The log likelihood values computed by the ``log_likelihood_function`` of each individual analysis class are summed to give an overall log likelihood value that the non-linear search samples when model-fitting.

- The output path structure of the results goes to a single folder, which includes sub-folders for the visualization of every individual analysis object based on the ``Analysis`` object's ``visualize`` method.

.. code-block:: python

    analysis = analysis_list[0] + analysis_list[1] + analysis_list[2]

We can alternatively sum a list of analysis objects as follows:

.. code-block:: python

    analysis = sum(analysis_list)

The ``log_likelihood_function``'s can be called in parallel over multiple cores by changing the ``n_cores`` parameter.

This is beneficial when the model-fitting procedure is slow and the likelihood evaluation time of the different
is roughly consistent.

.. code-block:: python

    analysis.n_cores = 1

To fit multiple datasets via a non-linear search we use this summed analysis object:

.. code-block:: python

    search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_simple")

    result_list = search.fit(model=model, analysis=analysis)

In the example above, the same ``Analysis`` class was used twice (to set up ``analysis_0`` and ``analysis_1``) and summed.

**PyAutoFit** supports the summing together of different ``Analysis`` classes, which may take as input completely different
datasets and fit the model to them (via the ``log_likelihood_function``) following a completely different procedure.

Result List
-----------

The result object returned by the fit is a list of the ``Result`` objects, which is described in the result cookbook.

Each ``Result`` in the list corresponds to each ``Analysis`` object in the ``analysis_list`` we passed to the fit.

The same model was fitted across all analyses, thus every ``Result`` in the ``result_list`` contains the same information 
on the samples and the same ``max_log_likelihood_instance``.

.. code-block:: python

    print(result_list[0].max_log_likelihood_instance.centre)
    print(result_list[0].max_log_likelihood_instance.normalization)
    print(result_list[0].max_log_likelihood_instance.sigma)

    print(result_list[1].max_log_likelihood_instance.centre)
    print(result_list[1].max_log_likelihood_instance.normalization)
    print(result_list[1].max_log_likelihood_instance.sigma)

This gives the following output:

.. code-block:: bash

    49.99110500540554
    24.793778321608457
    10.067848301502565
    49.99110500540554
    24.793778321608457
    10.067848301502565

We can plot the model-fit to each dataset by iterating over the results:

.. code-block:: python

    for data, result in zip(data_list, result_list):
        instance = result.max_log_likelihood_instance

        model_data = instance.model_data_1d_via_xvalues_from(
            xvalues=np.arange(data.shape[0])
        )

        plt.errorbar(
            x=xvalues,
            y=data,
            yerr=noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(xvalues, model_data, color="r")
        plt.title("Dynesty model fit to 1D Gaussian dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")
        plt.show()
        plt.close()

The image appears as follows:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_model_data_0.png
  :width: 300
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_model_data_1.png
  :width: 300
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_model_data_2.png
  :width: 300
  :alt: Alternative text

Variable Model Across Datasets
------------------------------

The same model was fitted to every dataset simultaneously because all 3 datasets contained an identical signal with 
only the noise varying across the datasets.

If the signal varied across the datasets, we would instead want to fit a different model to each dataset. The model
composition can be updated using the summed ``Analysis`` object to do this.

We will use an example of 3 1D Gaussians which have the same ``centre`` but the ``normalization`` and ``sigma`` vary across 
datasets:

.. code-block:: python

    dataset_path = path.join("dataset", "example_1d", "gaussian_x1_variable")

    dataset_name_list = ["sigma_0", "sigma_1", "sigma_2"]

    data_list = []
    noise_map_list = []

    for dataset_name in dataset_name_list:
        dataset_time_path = path.join(dataset_path, dataset_name)

        data = af.util.numpy_array_from_json(
            file_path=path.join(dataset_time_path, "data.json")
        )
        noise_map = af.util.numpy_array_from_json(
            file_path=path.join(dataset_time_path, "noise_map.json")
        )

        data_list.append(data)
        noise_map_list.append(noise_map)

Plotting these datasets shows that the ``normalization`` and`` ``sigma`` of each Gaussian vary.

.. code-block:: python

    for data, noise_map in zip(data_list, noise_map_list):
        xvalues = range(data.shape[0])

        af.ex.plot_profile_1d(xvalues=xvalues, profile_1d=data)

The images appear as follows:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_model_data_0.png
  :width: 300
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_model_data_1.png
  :width: 300
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/feature/docs_update/docs/images/multi_model_data_2.png
  :width: 300
  :alt: Alternative text


The ``centre`` of all three 1D Gaussians are the same in each dataset, but their ``normalization`` and ``sigma`` values 
are decreasing.

We will therefore fit a model to all three datasets simultaneously, whose ``centre`` is the same for all 3 datasets but
the ``normalization`` and ``sigma`` vary.

To do that, we use a summed list of ``Analysis`` objects, where each ``Analysis`` object contains a different dataset.

.. code-block:: python

    analysis_list = []

    for data, noise_map in zip(data_list, noise_map_list):
        analysis = af.ex.Analysis(data=data, noise_map=noise_map)
        analysis_list.append(analysis)

    analysis = sum(analysis_list)


We next compose a model of a 1D Gaussian.

.. code-block:: python

    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

We now update the model using the summed ``Analysis ``objects to compose a model where: 

- The ``centre`` values of the Gaussian fitted to every dataset in every ``Analysis`` object are identical.

- The``normalization`` and ``sigma`` value of the every Gaussian fitted to every dataset in every ``Analysis`` object are different.

The model has 7 free parameters in total, x1 shared ``centre``, x3 unique ``normalization``'s and x3 unique ``sigma``'s.

.. code-block:: python

    analysis = analysis.with_free_parameters(
        model.gaussian.normalization, model.gaussian.sigma
    )

To inspect this new model, with extra parameters for each dataset created, we extract a modified version of this 
model from the summed ``Analysis`` object.

This model modiciation occurs automatically when a non-linear search begins, therefore the normal model we created 
above is input to the ``search.fit()`` method.

.. code-block:: python

    model_updated = analysis.modify_model(model)

    print(model_updated.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 7

    model                                     Collection (N=7)
        0                                     Collection (N=3)
            gaussian                          Gaussian (N=3)
        1                                     Collection (N=3)
            gaussian                          Gaussian (N=3)
        2                                     Collection (N=3)
            gaussian                          Gaussian (N=3)

    0
        gaussian
            centre                            UniformPrior [7], lower_limit = 0.0, upper_limit = 100.0
            normalization                     LogUniformPrior [10], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                             UniformPrior [11], lower_limit = 0.0, upper_limit = 25.0
    1
        gaussian
            centre                            UniformPrior [7], lower_limit = 0.0, upper_limit = 100.0
            normalization                     LogUniformPrior [12], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                             UniformPrior [13], lower_limit = 0.0, upper_limit = 25.0
    2
        gaussian
            centre                            UniformPrior [7], lower_limit = 0.0, upper_limit = 100.0
            normalization                     LogUniformPrior [14], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma                             UniformPrior [15], lower_limit = 0.0, upper_limit = 25.0

Fit this model to the data using dynesty.

.. code-block:: python

    search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_free_sigma")


The ``normalization`` and ``sigma`` values of the maximum log likelihood models fitted to each dataset are different, 
which is shown by printing the ``sigma`` values of the maximum log likelihood instances of each result.

The ``centre`` values of the maximum log likelihood models fitted to each dataset are the same.

.. code-block:: python

    result_list = search.fit(model=model, analysis=analysis)

    for result in result_list:
        instance = result.max_log_likelihood_instance

        print("Max Log Likelihood Model:")
        print("Centre = ", instance.gaussian.centre)
        print("Normalization = ", instance.gaussian.normalization)
        print("Sigma = ", instance.gaussian.sigma)
        print()

This gives the following output:

.. code-block:: bash

    Max Log Likelihood Model:
    Centre =  50.03565394638727
    Normalization =  45.549160750232474
    Sigma =  24.99999730058904

    Max Log Likelihood Model:
    Centre =  50.03565394638727
    Normalization =  50.40062202023974
    Sigma =  20.28346578065846

    Max Log Likelihood Model:
    Centre =  50.03565394638727
    Normalization =  49.94394976751533
    Sigma =  9.98325143824908

Relational Model
----------------

In the model above, two extra free parameters (``normalization and ``sigma``) were added for every dataset. 

For just 3 datasets the model stays low dimensional and this is not a problem. However, for 30+ datasets the model
will become complex and difficult to fit.

In these circumstances, one can instead compose a model where the parameters vary smoothly across the datasets
via a user defined relation.

Below, we compose a model where the ``sigma`` value fitted to each dataset is computed according to:

.. code-block:: bash

    ``y = m * x + c`` : ``sigma`` = sigma_m * x + sigma_c``

Where x is an integer number specifying the index of the dataset (e.g. 1, 2 and 3).

By defining a relation of this form, ``sigma_m`` and ``sigma_c`` are the only free parameters of the model which vary
across the datasets. 

Of more datasets are added the number of model parameters therefore does not increase.

.. code-block:: python

    normalization_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    normalization_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    sigma_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    x_list = [1.0, 2.0, 3.0]

    analysis_with_relation_list = []

    for x, analysis in zip(x_list, analysis_list):
        normalization_relation = (normalization_m * x) + normalization_c
        sigma_relation = (sigma_m * x) + sigma_c

        analysis_with_relation = analysis.with_model(
            model.replacing(
                {
                    model.gaussian.normalization: normalization_relation,
                    model.gaussian.sigma: sigma_relation,
                }
            ),
        )

        analysis_with_relation_list.append(analysis_with_relation)

We can use division, subtraction and logorithms to create more complex relations and apply them to different parameters, 
for example:

.. code-block:: bash

    ``y = m * log10(x) - log(z) + c`` : ``sigma`` = sigma_m * log10(x) - log(z) + sigma_c``
    ``y = m * (x / z)`` : ``centre`` = centre_m * (x / z)``

.. code-block:: python

    model = af.Collection(gaussian=af.Model(af.ex.Gaussian))

    sigma_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    centre_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    x_list = [1.0, 10.0, 30.0]
    z_list = [2.0, 4.0, 6.0]

    analysis_with_relation_list = []

    for x, z, analysis in zip(x_list, z_list, analysis_list):
        sigma_relation = (sigma_m * af.Log10(x) - af.Log(z)) + sigma_c
        centre_relation = centre_m * (x / z)

        analysis_with_relation = analysis.with_model(
            model.replacing(
                {
                    model.gaussian.sigma: sigma_relation,
                    model.gaussian.centre: centre_relation,
                }
            )
        )

        analysis_with_relation_list.append(analysis_with_relation)

    analysis_with_relation = sum(analysis_with_relation_list)

Analysis summing is performed after the model relations have been created.

.. code-block:: python

    analysis_with_relation = sum(analysis_with_relation_list)

The modified model's ``info`` attribute shows the model has been composed using this relation.

.. code-block:: python

    model_updated = analysis.modify_model(model)

    print(model_updated.info)

This gives the following output:

.. code-block:: bash

    Total Free Parameters = 4
    
    model                                     Collection (N=4)
        0                                     Collection (N=4)
            gaussian                          Gaussian (N=4)
                centre                        MultiplePrior (N=1)
                sigma                         SumPrior (N=2)
                    self                      MultiplePrior (N=1)
                        other                 SumPrior (N=0)
                            result_value                                            Log10 (N=0)
                            other             Log (N=0)
        1                                     Collection (N=4)
            gaussian                          Gaussian (N=4)
                centre                        MultiplePrior (N=1)
                sigma                         SumPrior (N=2)
                    self                      MultiplePrior (N=1)
                        other                 SumPrior (N=0)
                            result_value                                            Log10 (N=0)
                            other             Log (N=0)
        2                                     Collection (N=4)
            gaussian                          Gaussian (N=4)
                centre                        MultiplePrior (N=1)
                sigma                         SumPrior (N=2)
                    self                      MultiplePrior (N=1)
                        other                 SumPrior (N=0)
                            result_value                                            Log10 (N=0)
                            other             Log (N=0)
    
    0
        gaussian
            centre
                centre_m                      UniformPrior [31], lower_limit = -0.1, upper_limit = 0.1
                other                         0.5
            normalization                     LogUniformPrior [27], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma
                sigma_c                       UniformPrior [30], lower_limit = -10.0, upper_limit = 10.0
                self
                    sigma_m                   UniformPrior [29], lower_limit = -0.1, upper_limit = 0.1
                    other
                        result_value
                            x                 1.0
                        other
                            z                 2.0
    1
        gaussian
            centre
                centre_m                      UniformPrior [31], lower_limit = -0.1, upper_limit = 0.1
                other                         2.5
            normalization                     LogUniformPrior [27], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma
                sigma_c                       UniformPrior [30], lower_limit = -10.0, upper_limit = 10.0
                self
                    sigma_m                   UniformPrior [29], lower_limit = -0.1, upper_limit = 0.1
                    other
                        result_value
                            x                 10.0
                        other
                            z                 4.0
    2
        gaussian
            centre
                centre_m                      UniformPrior [31], lower_limit = -0.1, upper_limit = 0.1
                other                         5.0
            normalization                     LogUniformPrior [27], lower_limit = 1e-06, upper_limit = 1000000.0
            sigma
                sigma_c                       UniformPrior [30], lower_limit = -10.0, upper_limit = 10.0
                self
                    sigma_m                   UniformPrior [29], lower_limit = -0.1, upper_limit = 0.1
                    other
                        result_value
                            x                 30.0
                        other
                            z                 6.0
We can fit the model as per usual.

.. code-block:: python

    search = af.DynestyStatic(path_prefix="features", name="multiple_datasets_relation")

    result_list = search.fit(model=model, analysis=analysis_with_relation)

The ``normalization`` and ``sigma`` values of the maximum log likelihood models fitted to each dataset are different, 
which is shown by printing the ``sigma`` values of the maximum log likelihood instances of each result.

They now follow the relation we defined above.

The ``centre`` values of the maximum log likelihood models fitted to each dataset are the same.

.. code-block:: python

    result_list = search.fit(model=model, analysis=analysis)

    for result in result_list:
        instance = result.max_log_likelihood_instance

        print("Max Log Likelihood Model:")
        print("Centre = ", instance.gaussian.centre)
        print("Normalization = ", instance.gaussian.normalization)
        print("Sigma = ", instance.gaussian.sigma)
        print()

This gives the following output:

.. code-block:: bash

    Max Log Likelihood Model:
    Centre =  50.03565394638727
    Normalization =  45.549160750232474
    Sigma =  24.99999730058904
    
    Max Log Likelihood Model:
    Centre =  50.03565394638727
    Normalization =  50.40062202023974
    Sigma =  20.28346578065846
    
    Max Log Likelihood Model:
    Centre =  50.03565394638727
    Normalization =  49.94394976751533
    Sigma =  9.98325143824908

Different Analysis Objects
--------------------------

For simplicity, this example summed together a single ``Analysis`` class which fitted 1D Gaussian's to 1D data.

For many problems one may have multiple datasets which are quite different in their format and structure In this 
situation, one can simply define unique ``Analysis`` objects for each type of dataset, which will contain a 
unique ``log_likelihood_function`` and methods for visualization.

The analysis summing API illustrated here can then be used to fit this large variety of datasets, noting that the 
the model can also be customized as necessary for fitting models to multiple datasets that are different in their 
format and structure. 

Interpolation
-------------

One may have many datasets which vary according to a smooth function, for example a dataset taken over time where
the signal varies smoothly as a function of time.

This could be fitted using the tools above, all at once. However, in many use cases this is not possible due to the
model complexity, number of datasets or computational time.

An alternative approach is to fit each dataset individually, and then interpolate the results over the smoothly
varying parameter (e.g. time) to estimate the model parameters at any point.

**PyAutoFit** has interpolation tools to do exactly this. These have not been documented yet, but if they sound
useful to you please contact us on SLACK and we'll be happy to explain how they work.

Individual Sequential Searches
------------------------------

The API above is used to create a model with free parameters across ``Analysis`` objects, which are all fit
simultaneously using a summed ``log_likelihood_function`` and single non-linear search.

Each ``Analysis`` can be fitted one-by-one, using a series of multiple non-linear searches, using
the ``fit_sequential`` method.

.. code-block:: python

    search = af.DynestyStatic(
        path_prefix="features", name="multiple_datasets_free_sigma__sequential"
    )

    result_list = search.fit_sequential(model=model, analysis=analysis)

The benefit of this method is for complex high dimensionality models (e.g. when many parameters are passed
to ``analysis.with_free_parameters``, it breaks the fit down into a series of lower dimensionality non-linear
searches that may convergence on a solution more reliably.

Hierarchical / Graphical Models

A common class of models used for fitting complex models to large datasets are hierarchical and graphical models. 

These models can include addition parameters not specific to individual datasets describing the overall 
relationship between different model components, thus allowing one to infer the global trends contained within a 
dataset.

**PyAutoFit** has a dedicated feature set for fitting hierarchical and graphical models and interested readers should
checkout the hierarchical and graphical modeling 
chapter of **HowToFit** (https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_graphical_models.html)

Wrap Up
--------

We have shown how **PyAutoFit** can fit large datasets simultaneously, using custom models that vary specific
parameters across the dataset.


