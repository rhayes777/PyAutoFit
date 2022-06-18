.. _multi_datasets:

Multiple Datasets
=================

**PyAutoFit** can easily perform model-fits to multiple datasets simultaneously, including tools to customize the model
composition that make it straight forward for specific parameters of the model to vary across the datasets.

This is achieve via summing of ``Analysis`` object, which you should recall each has its own unique dataset
and ``log_likelihood_function``. Thus, via analysis summing, we can perform model-fits to large and diverse datasets
where each individual dataset may only constrain specific aspects of a model.

This reflects the nature of temporal datasets which are accquired over time, where one can likely expect that certain
model parameters vary as a function of time (and therefore should be fitted to vary across the datasets) whilst others
remain fixed.

Identical Model Across Datasets
-------------------------------

Our first example dataset consists of three 1D Gaussian datasets.

All three datasets contain an identical signal, meaning that it is appropriate to fit the same model to all three
datasets simultaneously.

Each dataset has a different noise realization, meaning that performing a simultaneously fit will offer improved constraints
over individual fits.

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_0.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_1.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_2.png
  :width: 600
  :alt: Alternative text

Next, we create our model, which in this case corresponds to a single 1D Gaussian that will be fitted to all 3 datasets
simultaneously.

.. code-block:: python

    class Gaussian:

        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            normalization=0.1,  # <- constructor arguments are
            sigma=0.01,     # <- the Gaussian's parameters.
        ):

            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma

.. code-block:: python

    model = af.Collection(gaussian=af.Model(Gaussian))

We now set up our three instances of an ``Analysis`` class which includes the ``log_likelihood_function`` that fits a 1D Gaussian model
to each dataset and a ``visualize`` method for visualizing the fit.

.. code-block:: python

    analysis_list = []

    for data, noise_map in zip(data_list, noise_map_list):

        analysis = af.ex.Analysis(data=data, noise_map=noise_map)
        analysis_list.append(analysis)

We can now sum together every analysis in the list, to produce an overall analysis class which we fit with a non-linear
search.

By summing analysis objects the following happens:

 - The log likelihood values computed by the ``log_likelihood_function`` of each individual analysis class are summed to
 give the overall log likelihood value that the non-linear search uses for model-fitting.

 - The output path structure of the results goes to a single folder, which includes sub-folders for the visualization
 of every individual analysis object based on the ``Analysis`` object's ``visualize`` method.

.. code-block:: python

    analysis = analysis_list[0] + analysis_list[1] + analysis_list[2]

We can alternatively sum the analysis objects as follows:

.. code-block:: python

    analysis = sum(analysis_list)

The ``log_likelihood_function``'s can be called in parallel over multiple cores by changing the ``n_cores`` parameter:

.. code-block:: python

    analysis.n_cores = 1

To fit the multiple datasets via a non-linear search we use this analysis using the usual **PyAutoFit** API:

.. code-block:: python

    search = af.DynestyStatic(path_prefix=path.join("features"), name="multiple_datasets_simple")

    result_list = search.fit(model=model, analysis=analysis)

The result object returned by the fit is a list of the ``Result`` objects you have used in other examples.

In this example, the same model is fitted across all analyses, thus every ``Result`` in the ``result_list`` contains
the same information on the samples and thus gives the same output from methods such
as ``max_log_likelihood_instance``.

.. code-block:: python

    print(result_list[0].max_log_likelihood_instance)
    print(result_list[1].max_log_likelihood_instance)

Inspection of the results show tht the model was successfully fitted to all three datasets:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_model_0.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_model_1.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_model_2.png
  :width: 600
  :alt: Alternative text


Variable Model Across Datasets
------------------------------

Above, the same model was fitted to every dataset simultaneously, which was possible because all 3 datasets contained
an identical signal with only the noise varying across the datasets.

It is common for the signal in each dataset to be different and for it to constrain only certain aspects of the model.
The model parameterization therefore needs to change in order to account for this.

Lets look at an example of a dataset of 3 1D Gaussians where the signal varies across the datasets:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_model_vary_0.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_model_vary_1.png
  :width: 600
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/gaussian_model_vary_2.png
  :width: 600
  :alt: Alternative text

In this case, the ``centre`` and ``normalization`` of all three 1D Gaussians are the same in each dataset,
but their ``sigma`` values are decreasing.

We therefore wish to compose and to fit a model to all three datasets simultaneously, where the ``centre``
and ``normalization`` are the same across all three datasets but the ``sigma`` value is unique for each dataset.

To do that, we interface a model with a summed list of analysis objects

.. code-block:: python

    model = af.Collection(gaussian=af.Model(Gaussian))

    analysis = analysis.with_free_parameters(
        *[model.gaussian.sigma]
    )

We code above updates the model using the summed ``Analysis ``objects to compose a model where:

 - The ``centre`` and ``normalization`` values of the Gaussian fitted to every dataset in every ``Analysis`` object are
 identical.

 - The ``sigma`` value of the every Gaussian fitted to every dataset in every ``Analysis`` object are different.

This means that the model has 5 free parameters in total, the shared ``centre`` and ``normalization`` and a unique
``sigma`` value for every dataset.

We can again fit this model as per usual:

.. code-block:: python

    search = af.DynestyStatic(path_prefix=path.join("features"), name="multiple_datasets_free_sigma")

    result_list = search.fit(model=model, analysis=analysis)

Variable Model With Relationships
---------------------------------

In the model above, an extra free parameter ``sigma`` was added for every dataset.

This was ok for the simple model fitted here to just 3 datasets, but for more complex models and problems with 10+
datasets one will quick find that the model complexity increases dramatically.

In these circumstances, one can instead compose a model where the parameters vary smoothly across the datasets
via a user defined relation.

Below, we compose a model where the ``sigma`` value fitted to each dataset is computed according to:

 ``y = m * x + c`` : ``sigma`` = sigma_m * x + sigma_c``

Where x is an integer number specifying the index of the dataset (e.g. 1, 2 and 3).

By defining a relation of this form, ``sigma_m`` and ``sigma_c`` are the only free parameters of the model which vary
across the datasets.

Therefore, if more datasets are added the number of model parameter does not increase, like we saw above.

.. code-block:: python

    sigma_m = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    sigma_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    x_list = [1.0, 2.0, 3.0]

    analysis_with_relation_list = []

    for x, analysis in zip(x_list, analysis_list):

        sigma_relation = (sigma_m * x) + sigma_c

        analysis_with_relation = analysis.with_model(
                model.replacing(
                    {
                        model.gaussian.sigma:sigma_relation
                    }
                )
            )

        analysis_with_relation_list.append(analysis_with_relation)


We can fit this model as per usual, you may wish to checkout the ``model.info`` file to see how a schematic of this
model's composition.

.. code-block:: python

    analysis_with_relation = sum(analysis_with_relation_list)

    search = af.DynestyStatic(path_prefix=path.join("features"), name="multiple_datasets_relation")

    result_list = search.fit(model=model, analysis=analysis_with_relation)

Temporally Varying Models
-------------------------

An obvious example of fitting models which vary across datasets are time-varying models, where the datasets are
observations of a signal which varies across time.

In such circumstances, it is common for certain model parameters to be known to not vary as a function of time (and
therefore be fixed across the datasets) whereas other parameters are known to vary as a function of time (and therefore
should be parameterized accordingly using the API illustrated here).

Different Analysis Objects
--------------------------

For simplicity, this example summed together only a single ``Analysis`` class.

For many problems one may have multiple datasets which are quite different in their format and structure (perhaps
one is a  1D signal whereas another dataset is an image). In this situation, one can simply define unique ``Analysis``
objects for each type of dataset, which will contain a unique ``log_likelihood_function`` and methods for visualization.

Nevertheless, the analysis summing API illustrated here will still work, meaning that **PyAutoFit** makes it simple to
fit highly customized models to multiple datasets that are different in their format and structure.

Graphical Models
----------------

A common class of models used for fitting complex models to large datasets are graphical models.

Graphical models can include addition parameters not specific to individual datasets describing the overall
relationship between different model components, thus allowing one to infer the global trends contained within a
dataset.

**PyAutoFit** has a dedicated feature set for fitting graphical models and interested readers should
checkout the `graphical modeling chapter of **HowToFit** <https://pyautofit.readthedocs.io/en/latest/howtofit/chapter_graphical_models.html>`_.

Wrap-Up
-------

We have shown how **PyAutoFit** can fit large datasets simultanoeusly, using custom models that vary specific
parameters across the dataset.

The ``autofit_workspace/*/model/cookbook_3_multiple_datasets`` cookbook and following readthedocs page give a concise
API reference for model composition when fitting multiple datasets.