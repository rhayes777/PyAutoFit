import autofit as af


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


def test_add_analysis():
    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2
