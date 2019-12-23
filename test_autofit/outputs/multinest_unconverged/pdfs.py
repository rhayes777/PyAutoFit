from pathlib import Path

import autofit as af
from autofit.optimize.non_linear.multi_nest import MultiNestOutput

test_path = Path(__file__).parent.parent / "multinest_converged/"
output_path = test_path / "output/"

af.conf.instance = af.conf.Config(
    config_path=str(test_path / "config"), output_path=str(output_path)
)

pipeline_name = "pipeline_main__x1_gaussian"
phase_name = "phase_1__x1_gaussian_final"

aggregator = af.Aggregator(directory=str(output_path))

multinest_outputs = aggregator.filter(pipeline=pipeline_name, phase=phase_name).output

print(multinest_outputs)

## TODO : This should be handled by the aggregator

multinest_outputs = [
    MultiNestOutput(model=output.model, paths=output.paths)
    for output in multinest_outputs
]

multinest_outputs[0].output_pdf_plots()
