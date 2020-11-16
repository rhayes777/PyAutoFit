.. _pipelines:

Transdimensional Pipelines
==========================

In **PyAutoFit**, a ``Pipeline`` is a sequence of ``NonLinearSearch``'s (or ``phase``'s) which fit different
models to a dataset. Initial phases fit simplified realizations of the model whose results are used to
initialize model-fits using more complex models in later phases.

We call this a **transdimensional model-fitting pipelines**, and they are great for:

 - Making the fitting of complex models **fully automated**, by breaking the model-fitting procedure down
   into a series of simplier *bite-sized* model fits.
 - Fitting many different models to a dataset to streamline Bayesian model comparison.
 - Augmenting the data and and adapting the model-fitting procedure between phases, exploiting the knowledge
   gained in earlier phases to improve the analysis performed in later phases.

To illustrate this, lets extend our example of fitting 1D ``Gaussian``'s to a problem with two ``Gaussian``'s:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/toy_model_fit.png
  :width: 400
  :alt: Alternative text

  https://github.com/rhayes777/PyAutoFit/blob/master/docs/advanced/images/gaussian_x2_split.png

The data has two distinguishable ``Gaussian``'s, one on the left and one on the right. Traditional
approaches may fit both ``Gaussian``'s simultaneously, making parameter space more complex, slower
to sample and increasing the risk that we fail to locate the global maxima solution.

A ``Pipeline``, can break the model-fit down into 3 phases:

1) Fit only the left ``Gaussian``.
2) Fit only the right ``Gaussian``, using the model of the left ``Gaussian`` from phase 1 to reduce blending.
3) Fit both simultaneously, using the results of phase 1 & 2 to initialize where the ``NonLinearSearch``
   samples parameter space.

Lets look at an example:

.. code-block:: python

    def make_pipeline():

        pipeline_name = "pipeline__x2_gaussians"

        """
        Phase 1:

        Fit the Gaussian on the left by restricting the centre of its profile to the first 50
        pixels and removing the right half of the data.
        """

        gaussian_0 = af.PriorModel(profiles.Gaussian)
        gaussian_0.add_assertion(gaussian_0.centre < 50)

        phase1 = ph.Phase(
            name="phase_1__left_gaussian",
            folders=folders,
            profiles=af.CollectionPriorModel(gaussian_0=gaussian_0),
            search=af.PySwarmsGlobal()
        )

        """
        Phase 2:

        Fit the Gaussian on the right, by restricting the centre of its profile to the last 50
        pixels and removing the left half the data.

        The best-fit Gaussian resulting from phase 1 above is used to fit the left-hand Gaussian.
        """

        gaussian_1 = af.PriorModel(profiles.Gaussian)
        gaussian_1.add_assertion(gaussian_1.centre > 50)

        phase2 = ph.Phase(
            name="phase_2__right_gaussian",
            folders=folders,
            profiles=af.CollectionPriorModel(
                gaussian_0=phase1.result.instance.profiles.gaussian_0,  # <- phase1 Gaussian.
                gaussian_1=gaussian_1,
            ),
            search=af.PySwarmsGlobal()
        )

        """
        Phase 3:

        Fit both Gaussians to the full dataset, using the results of phases 1 and 2 to
        initialize the model parameters.
        """

        phase3 = ph.Phase(
            name="phase_3__both_gaussian",
            folders=folders,
            profiles=af.CollectionPriorModel(
                gaussian_0=phase1.result.model.profiles.gaussian_0,  # <- phase1 Gaussian.
                gaussian_1=phase2.result.model.profiles.gaussian_1,  # <- phase2 Gaussian.
            ),
            search=af.DynestyStatic()
        )

        return Pipeline(pipeline_name, phase1, phase2, phase3)

The resulting model-fits of phases 1, 2 and 3 are shown below:

![alt text](https://github.com/rhayes777/PyAutoFit/blob/master/docs/advanced/images/gaussian_x2_split.png)

In the first two phases we only required a 1D ``Gaussian`` that fitted its half of the data
*reasonably well*, so we used the `search` `PySwarms`. This samples parameter space but doesn't infer
errors on the model parameters, which is fine given all we wanted was an initialization for phase 3!

In phase 3, we required a fit of the full model complete *robust* error estimation, therefore we switched
to the nested sampler ``Dynesty``. This takes longer to fit the model, but gains a significant run-time
boost by using the information passed from phases 1 and 2 to begin sampling the more complex parameter
space in the higher likelihood regions!

``Pipelines`` built in this way exploit **domain specific knowledge**. We are using our understanding of the model
fitting task (that the data contains two ``Gaussian``'s split on the left and right hand sides) to perform a
more efficent and robust model-fit.

Although illustrative, the example above is somewhat trivial. However, using ``Pipeline``'s to exploit
**domain specific knowledge** has proven crucial for our child project
`PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_, an Astronomy package that fits complex images of
distant galaxies.
This `example pipeline <https://github.com/Jammy2211/autolens_workspace/blob/master/transdimensional/pipelines/imaging/light_dark/light_bulge_disk__mass_mlr_dark__source_inversion.py>`_
fits a 28 parameter model of a galaxies light by breaking the fit of the model in different region of the
image into 5 distinct phases, switching the ``NonLinearSearch`` between phases and augmenting the data to
speed up the fit in earlier phases.

If ``Pipeline``'s suit your model-fitting problem, checkout the tutorials in chapter 3 of the **HowToFit**
lectures. These explain how to implement the functionality in your source code and advanced pipeline features
not covered here!
