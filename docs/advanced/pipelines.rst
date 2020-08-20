.. _pipelines:

Transdimensional Pipelines
==========================

In **PyAutoFit**, a pipeline is a sequence of non-linear searches (or *Phase*'s) which fit different models to a
dataset. The different *Phase*'s compose and fit many different models, whereby the initial phases fit simplified
realizations of the model whose results are used to initialize model-fits using more complex models in later phases.

We call these *Pipeline*'s objects **transdimensional model-fitting pipelines**, and they allow:

 - The fitting of complex models to be **fully automated**, by breaking the model-fitting procedure down into a series
   of simplier *bite-sized* model fits.
 - Many different models to be fitted to data and compared with one another in a computationally efficient manner.
 - The data and model-fitting procedure to be augmented and adapted between phases, exploiting the knowledge gained in
   earlier phases to improve the fit.

To illustrate this, we'll extend our example of fitting 1D Gaussians to a problem where there are two Gaussians in the
data:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/toy_model_fit.png
  :width: 400
  :alt: Alternative text

  https://github.com/rhayes777/PyAutoFit/blob/master/docs/advanced/images/gaussian_x2_split.png

The data has two Gaussians which clearly distinguishable, one is on the left hand side and the other is on the right
hand side. Traditional approaches would fit both Gaussians simultaneously, making parameter space more complex, slower
to sample and increasing the risk that we fail to locate the global maxima solution.

By writing a *Pipeline*, we can break the model-fit down into 3 phases:

1) Fit only the left Gaussian.
2) Fit only the right Gaussian, using the model of the left Gaussian from phase 1 to reduce blending.
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear search
   searches parameter space.

.. code-block:: python

    def make_pipeline(folders=None):

        if folders is None:
            folders = []

        pipeline_name = "pipeline__x2_gaussians"

        setup.folders.append(pipeline_name)

        """
        Phase 1:

        Fit the Gaussian on the left by restricting the centre of its profile to the first 50 pixels and removing the
        right half of the data.
        """

        gaussian_0 = af.PriorModel(profiles.Gaussian)
        gaussian_0.add_assertion(gaussian_0.centre < 50)

        phase1 = ph.Phase(
            phase_name="phase_1__left_gaussian",
            folders=folders,
            profiles=af.CollectionPriorModel(gaussian_0=gaussian_0),
            settings=SettingsPhase(trim_data_left=50), # Remove the right-hand side of the data.
            search=af.DynestyStatic(), # Use an optimizer for fast non-linear sampling.
        )

        """
        Phase 2:

        Fit the Gaussian on the right, by restricting the centre of its profile to the last 50 pixels and removing the
        left half the data.

        The best-fit Gaussian resulting from phase 1 above is used to fit the left-hand Gaussian.
        """

        gaussian_1 = af.PriorModel(profiles.Gaussian)
        gaussian_1.add_assertion(gaussian_1.centre > 50)

        phase2 = ph.Phase(
            phase_name="phase_2__right_gaussian",
            folders=folders,
            profiles=af.CollectionPriorModel(
                gaussian_0=phase1.result.instance.profiles.gaussian_0,  # <- Use the Gaussian fitted in phase 1
                gaussian_1=gaussian_1,
            ),
            settings=SettingsPhase(trim_data_right=50), # Remove the left-hand side of the data.
            search=af.PySwarms(), # Use an optimizer for fast non-linear sampling.
        )

        """
        Phase 3:

        Fit both Gaussians to the full dataset, using the results of phases 1 and 2 to initialize the model parameters.
        """

        phase3 = ph.Phase(
            phase_name="phase_3__both_gaussian",
            folders=folders,
            profiles=af.CollectionPriorModel(
                gaussian_0=phase1.result.model.profiles.gaussian_0,  # <- use phase 1 Gaussian results.
                gaussian_1=phase2.result.model.profiles.gaussian_1,  # <- use phase 2 Gaussian results.
            ),
            search=af.DynestyStatic(), # Use a nested sampler for robust error estimation.
        )

        return Pipeline(pipeline_name, phase1, phase2, phase3)

The resulting model-fits of phases 1, 2 and 3 are shown below:

![alt text](https://github.com/rhayes777/PyAutoFit/blob/master/docs/advanced/images/gaussian_x2_split.png)

In the first two phases we only required a 1D Gaussian that fitted their half of the data *reasonably well*, to act as
initialization for phase 3. Therefore, we first trimmed the half of the data we were not fitting, speeding up the
model-fitting process. These phases also used the *PySwarms* optimizer to fit the model, a non-linear search which
quickly maximizes the fit likelihood (but does not provide model error estimates).

In phase 3, we want a *robust* fit to the complete dataset with model error estimation, therefore we did not trim the
data and switched to the nested sampler *Dynesty*. This used the information provided to it by phases 1 and 2 to
more quickly and accurately sample the more complex parameter space that includes both 1D Gaussians.

Here, we are exploiting **domain specific knowledge** to perform a more efficent and robust model-fit. We are using our
knowledge of the problem (e.g. that there are two Gaussians in the data that are split on the left and right hand side)
to adapt and improve our model-fitting procedure to the task at hand.

Although this illustrative example is somewhat trivial, using *Pipeline*'s to exploit **domain specific knowledge**
has proven crucial for the project `PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_, which fits images of
gravitationally lensed galaxies. This example pipeline fits a complex 28 parameter model for a galaxies light and
mass distributions by breaking the model-fit down into 5 distinct phases - a model we would be unable to fit in a
**fully automated** manner using just one non-linear search!

If you think the use of *Pipeline*'s suits you model-fitting problem, we recommend you checkout the relevant tutorials
in chapter 2 of the **HowToFit** lectures. These explain how to implement the functionality in your source code and
advanced pipeline features not covered here!
