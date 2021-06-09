import autofit as af
from autoconf import conf


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


def test_add_analysis():
    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2


def test_two_cores():
    conf.instance["general"]["analysis"]["n_cores"] = 2

    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2
