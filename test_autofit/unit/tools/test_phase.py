import autofit as af

def test__phase_settings_tag():

    settings = af.AbstractPhaseSettings(log_likelihood_cap=None)
    assert settings.log_likelihood_cap_tag == ""

    settings = af.AbstractPhaseSettings(log_likelihood_cap=10.0)
    assert settings.log_likelihood_cap_tag == "__lh_cap_10.0"

    settings = af.AbstractPhaseSettings(log_likelihood_cap=200.0001)
    assert settings.log_likelihood_cap_tag == "__lh_cap_200.0"