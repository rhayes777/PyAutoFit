.. _pipelines:

Pipelines
=========

In transdimensional modeling many different models are paramertized and fitted to the same data-set.

This is performed using **transdimensional model-fitting pipelines**, which break the model-fit into a series of
**linked non-linear searches**, or phases. Initial phases fit simplified realizations of the model, whose results are
used to initialize fits using more complex models in later phases.

Fits of complex models with large dimensionality can therefore be broken down into a series of
**bite-sized model fits**, allowing even the most complex model fitting problem to be **fully automated**.

Lets illustrate this with an example fitting two 2D Gaussians:

![alt text](https://github.com/rhayes777/PyAutoFit/blob/master/gaussian_example.png)

We're going to fit each with the 2D Gaussian profile above. Traditional approaches would fit both Gaussians
simultaneously, making parameter space more complex, slower to sample and increasing the risk that we fail to locate
the global maxima solution. With **PyAutoFit** we can instead build a transdimensional model fitting pipeline which
breaks the the analysis down into 3 phases:

1) Fit only the left Gaussian.
2) Fit only the right Gaussian, using the model of the left Gaussian from phase 1 to reduce blending.
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear search
   searches parameter space.

.. code-block:: python

    def make_pipeline():

        # In phase 1, we will fit the Gaussian on the left.

        phase1 = af.Phase(
            phase_name="phase_1__left_gaussian",
            gaussians=af.CollectionPriorModel(gaussian_0=Gaussian),
            non_linear_class=af.MultiNest,
        )

        # In phase 2, we will fit the Gaussian on the right, where the best-fit Gaussian
        # resulting from phase 1 above fits the left-hand Gaussian.

        phase2 = af.Phase(
            phase_name="phase_2__right_gaussian",
            phase_folders=phase_folders,
            gaussians=af.CollectionPriorModel(
                # Use the Gaussian fitted in Phase 1:
                gaussian_0=phase1.result.instance.gaussians.gaussian_0,
                gaussian_1=Gaussian,
            ),
            non_linear_class=af.MultiNest,
        )

        # In phase 3, we fit both Gaussians, using the results of phases 1 and 2 to
        # initialize their model parameters.

        phase3 = af.Phase(
            phase_name="phase_3__both_gaussian",
            phase_folders=phase_folders,
            gaussians=af.CollectionPriorModel(
                # use phase 1 Gaussian results as priors.
                gaussian_0=phase1.result.model.gaussians.gaussian_0,
                # use phase 2 Gaussian results as priors.
                gaussian_1=phase2.result.model.gaussians.gaussian_1,
            ),
            non_linear_class=af.MultiNest,
        )

        return toy.Pipeline(pipeline_name, phase1, phase2, phase3)

`PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_ shows a real-use case of transdimensional modeling, fitting
galaxy-scale strong gravitational lenses. In this example pipeline, a 5-phase **PyAutoFit** pipeline breaks-down the
fit of 5 diferent models composed of over 10 unique model components and 10-30 free parameters.