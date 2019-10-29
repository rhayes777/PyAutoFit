import autofit as af
from test_autofit.integration.tests import runner

test_type = "gaussian"
test_name = "phase_x1"
data_type = "gaussian"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    phase1 = af.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return af.PipelineImaging(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
