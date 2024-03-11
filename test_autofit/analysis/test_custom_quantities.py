import autofit as af
from autofit.non_linear.analysis.custom_quantities import CustomQuantities


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        self.save_custom_quantities(centre=instance.centre)
        return 1.0


def test_custom_quantities():
    custom_quantities = CustomQuantities()
    custom_quantities.add(centre=1.0)

    assert custom_quantities.names == ["centre"]
    assert custom_quantities.values == [[1.0]]


def test_analysis_custom_quantities():
    analysis = Analysis()
    instance = af.Gaussian()
    analysis.log_likelihood_function(instance=instance)

    assert analysis.custom_quantities == [{"centre": instance.centre}]
