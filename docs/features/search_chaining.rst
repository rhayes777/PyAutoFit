.. _search_chaining:

Search Chaining
---------------

To perform a model-fit, we typically compose one model and fit it to our data using one non-linear search.

Search chaining fits many different models to a dataset using a chained sequence of non-linear searches. Initial
fits are performed using simplified model parameterizations and faster non-linear fitting techniques. The results of
these simplified fits can then be used to initialize fits using a higher dimensionality model with more detailed
non-linear search.

To fit highly complex models our aim is therefore to **granularize** the fitting procedure into a series of **bite-sized**
searches which are faster and more reliable than fitting the more complex model straight away.

Our ability to construct chained non-linear searches that perform model fitting more accurately and efficiently relies
on our **domain specific knowledge** of the model fitting task. For example, we may know that our dataset contains
multiple features that can be accurately fitted separately before performing a joint fit, or that certain parameters
share minimal covariance such that allowing us to fix them to certain values in earlier model-fits.

We may also know tricks that can speed up the fitting of the initial model, for example reducing the size of the data
or changing making a likelihood evaluation faster (most likely at the expense of the quality of the fit itself). By
using chained searches speed-ups can be relaxed towards the end of the model-fitting sequence when we want the most
precise, most accurate model that best fits the dataset available.

In this example we demonstrate search chaining using the example data where there are two ``Gaussians`` that are visibly
split:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x2_split.png
  :width: 600
  :alt: Alternative text

Instead of fitting them simultaneously using a single non-linear search consisting of N=6 parameters, we break
this model-fit into a chained of three searches where:

1) The first model fits just the left ``Gaussian`` where N=3.
2) The second model fits just the right ``Gaussian`` where again N=3.
3) The final model is fitted with both ``Gaussians`` where N=6. Crucially, the results of the first two searches
are used to initialize the search and tell it the highest likelihood regions of parameter space.

By initially fitting parameter spaces of reduced complexity we can achieve a more efficient and reliable model-fitting
procedure.

To fit the left ``Gaussian``, our first ``analysis`` receive only half data removing the right ``Gaussian``. Note that
this give a speed-up in log likelihood evaluation.

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x2_left.png
  :width: 600
  :alt: Alternative text

We now create a search to fit this data. Given the simplicity of the model, we can use a low number of live points
to achieve a fast model-fit (had we fitted the more complex model right away we could not of done this).

.. code-block:: bash

    model = af.Collection(gaussian_left=m.Gaussian)

    dynesty = af.DynestyStatic(
        name=("search[1]__left_gaussian"),
        nlive=30,
    )

    search_2_result = dynesty.fit(model=model, analysis=analysis)

By plotting the result we can see we have fitted the left ``Gaussian`` reasonably well.

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x2_left_fit.png
  :width: 600
  :alt: Alternative text

We now repeat the above process for the right ``Gaussian``.

We could remove the data on the left like we did the ``Gaussian`` above. However, we are instead going to fit the full
dataset. To fit the left Gaussian we use the maximum log likelihood model of the model inferred in search 1.

For search chaining, **PyAutoFit** has many convenient methods for passing the results of a search to a subsequence
search. Below, we achieve this by passing the result of the search above as an ``instance.

.. code-block:: bash

    model = af.Collection(
        gaussian_left=search_1_result.instance.gaussian_left,
        gaussian_right=m.Gaussian
    )

We now run our second Dynesty search to fit the right ``Gaussian``. We can again exploit the simplicity of the model
and use a low number of live points to achieve a fast model-fit.

.. code-block:: bash

    dynesty = af.DynestyStatic(
        name=("search[2]__right_gaussian"),
        path_prefix=path.join("features", "search_chaining"),
        nlive=30,
        iterations_per_update=500,
    )

search_2_result = dynesty.fit(model=model, analysis=analysis)

We can now see our model has successfully fitted both Gaussians:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x2_right_fit.png
  :width: 600
  :alt: Alternative text

We now fit both ``Gaussians``'s simultaneously, using the results of the previous two searches to initialize where
the non-linear searches parameter space.

To pass the result in this way we use the command ``result.model``, which in contrast to ``result.instance`` above passes
the parameters not as the maximum log likelihood values but as ``GaussianPrior``'s that are fitted for by the
non-linear search.

The ``mean`` and ``sigma`` value of each parmeter's ``GaussianPrior`` are set using the results of searches 1 and
2 to ensure our model-fit only searches the high likelihood regions of parameter space.

.. code-block:: bash

    model = af.Collection(
        gaussian_left=search_1_result.model.gaussian_left,
        gaussian_right=search_2_result.model.gaussian_right
    )

    dynesty = af.DynestyStatic(
        name=("search[3]__both_gaussians"),
        path_prefix=path.join("features", "search_chaining"),
        nlive=100,
        iterations_per_update=500,
    )

    search_3_result = dynesty.fit(model=model, analysis=analysis)

We can now see our model has successfully fitted both Gaussians simultaneously:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/features/images/gaussian_x2_fit.png
  :width: 600
  :alt: Alternative text

This fit used a technique called 'prior passing' to pass results from searches 1 and 2 to search 3. Full details of how
prior passing works can be found in the ``search_chaining.ipynb`` feature notebook.