import autofit as af
from autoconf.conf import with_config


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1


def test_add_analysis():
    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2


@with_config(
    "general", "analysis", "n_cores",
    value=2
)
def test_two_cores():
    assert (Analysis() + Analysis()).log_likelihood_function(
        None
    ) == -2


def test_still_flat():
    analysis = (Analysis() + Analysis()) + Analysis()

    assert len(analysis) == 3

    analysis = Analysis() + (Analysis() + Analysis())

    assert len(analysis) == 3
