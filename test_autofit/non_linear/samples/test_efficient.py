import autofit as af
from autofit.non_linear.samples.efficient import EfficientSamples


def test():
    samples = af.SamplesPDF(
        model=af.Model(af.Gaussian),
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=4.0,
                kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0},
            )
        ],
    )
    efficient = EfficientSamples(samples=samples)
    samples = efficient.samples

    sample = samples.sample_list[0]
    assert sample.log_likelihood == 1.0
    assert sample.log_prior == 2.0
    assert sample.weight == 4.0
