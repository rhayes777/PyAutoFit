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

print(multinest_outputs)

most_likely_vector = [output.most_likely_vector for output in multinest_outputs]

print(most_likely_vector)

most_probable_vector = [output.most_probable_vector for output in multinest_outputs]

print(most_probable_vector)

most_probable_instances = [
    output.most_probable_instance for output in multinest_outputs
]

print(most_probable_instances)

model_parameters_at_1_sigma = [
    output.vector_at_sigma(sigma=1.0) for output in multinest_outputs
]

print(model_parameters_at_1_sigma)

[output.output_results(during_analysis=False) for output in multinest_outputs]
