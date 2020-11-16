import autofit as af


class Pipeline(af.Pipeline):
    def run(self, dataset, mask, info=None):
        def runner(phase, results):
            return phase.run(dataset=dataset, results=results, mask=mask, info=info)

        return self.run_function(runner)


from howtofit.chapter_phase_api_non_linear_searches.src.phase import (
    phase as ph,
)
from howtofit.chapter_phase_api_non_linear_searches.src.model import (
    profiles,
)


def make_pipeline(folders=None):

    if folders is None:
        folders = []

    pipeline_name = "pipeline__x2_gaussians"

    setup.path_prefix.append(pipeline_name)

    """
    Phase 1:
     
    Fit the `Gaussian` on the left by restricting the centre of its profile to the first 30 pixels.  
    """

    gaussian_0 = af.PriorModel(profiles.Gaussian)
    gaussian_0.add_assertion(gaussian_0.centre < 30)

    phase1 = ph.Phase(
        name="phase_1__left_gaussian",
        folders=folders,
        profiles=af.CollectionPriorModel(gaussian_0=gaussian_0),
        search=af.DynestyStatic(),
    )

    """
    Phase 2: 
    
    Fit the `Gaussian` on the right, by restricting the centre of its profile to the last 30 pixels.   
    
    The best-fit `Gaussian` resulting from phase 1 above is used to fit the left-hand Gaussian.
    """

    gaussian_1 = af.PriorModel(profiles.Gaussian)
    gaussian_1.add_assertion(gaussian_1.centre > 70)

    phase2 = ph.Phase(
        name="phase_2__right_gaussian",
        folders=folders,
        profiles=af.CollectionPriorModel(
            gaussian_0=phase1.result.instance.profiles.gaussian_0,  # <- Use the `Gaussian` fitted in phase 1
            gaussian_1=gaussian_1,
        ),
        search=af.DynestyStatic(),
    )

    """
    Phase 3:
     
    Fit both Gaussians, using the results of phases 1 and 2 to initialize their model parameters.
    """

    phase3 = ph.Phase(
        name="phase_3__both_gaussian",
        folders=folders,
        profiles=af.CollectionPriorModel(
            gaussian_0=phase1.result.model.profiles.gaussian_0,  # <- use phase 1 `Gaussian` results.
            gaussian_1=phase2.result.model.profiles.gaussian_1,  # <- use phase 2 `Gaussian` results.
        ),
        search=af.DynestyStatic(),
    )

    return Pipeline(pipeline_name, phase1, phase2, phase3)
