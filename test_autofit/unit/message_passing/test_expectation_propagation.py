import pytest
from scipy import stats

from autofit import message_passing as mp


@pytest.fixture(
    name="normal_factor"
)
def make_normal_factor(x):
    return mp.factor(
        stats.norm(
            loc=-0.5,
            scale=0.5
        ).logpdf
    )(x)


@pytest.fixture(
    name="model_approx"
)
def make_model_approx(
        probit_factor,
        normal_factor
):
    model = probit_factor * normal_factor
    return mp.MeanFieldApproximation.from_kws(
        model,
        x=mp.NormalMessage(0, 1)
    )


@pytest.fixture(
    name="probit_approx"
)
def make_probit_approx(
        probit_factor,
        model_approx
):
    return model_approx.factor_approximation(
        probit_factor
    )


def test_approximations(
        probit_approx
):
    assert isinstance(
        probit_approx,
        mp.FactorApproximation
    )
