import numpy as np
import pytest
from scipy import stats

import autofit.graphical.messages.normal
from autofit import graphical as mp


@pytest.fixture(
    name="normal_factor"
)
def make_normal_factor(x):
    return mp.Factor(
        stats.norm(
            loc=-0.5,
            scale=0.5
        ).logpdf,
        x=x
    )


@pytest.fixture(
    name="model"
)
def make_model(
        probit_factor,
        normal_factor
):
    return probit_factor * normal_factor


@pytest.fixture(
    name="model_approx"
)
def make_model_approx(
        model,
        x
):
    return mp.EPMeanField.from_kws(
        model,
        {x: autofit.graphical.messages.normal.NormalMessage(0, 1)}
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
        probit_approx,
        model_approx,
        x
):
    opt_probit = mp.OptFactor.from_approx(probit_approx)
    result = opt_probit.maximise({x: 0.})

    probit_model = autofit.graphical.messages.normal.NormalMessage.from_mode(
        result.mode[x],
        covariance=result.hess_inv[x]
    )

    probit_model_dist = mp.MeanField({x: probit_model})

    # get updated factor approximation
    probit_project, status = probit_approx.project(
        probit_model_dist, delta=1.
    )

    assert probit_project.model_dist[x].mu == pytest.approx(0.506, rel=0.1)
    assert probit_project.model_dist[x].sigma == pytest.approx(0.814, rel=0.1)

    assert probit_project.factor_dist[x].mu == pytest.approx(1.499, rel=0.1)
    assert probit_project.factor_dist[x].sigma == pytest.approx(1.401, rel=0.1)


def test_looped_importance_sampling(
        model,
        normal_factor,
        probit_factor,
        x
):
    model_approx = mp.EPMeanField.from_kws(
        model,
        {x: autofit.graphical.messages.normal.NormalMessage(0, 1)}
    )

    np.random.seed(1)
    sampler = mp.ImportanceSampler(
        n_samples=1000
    )
    history = list()

    for i in range(3):
        for factor in [normal_factor, probit_factor]:
            # get factor, cavity distribution and model approximation
            factor_approx = model_approx.factor_approximation(factor)
            # sample from model approximation
            sample = sampler.sample(factor_approx)
            # project sufficient statistcs of sample onto normal dist
            model_dist = mp.project_factor_approx_sample(
                factor_approx,
                sample
            )

            # divide projection by cavity distribution
            factor_project, status = factor_approx.project(
                model_dist,
                delta=1
            )

            # update model approximation
            model_approx, status = model_approx.project(
                factor_project,
                status
            )

            # save and print current approximation
            history.append(model_approx)
            

    result = history[-1].mean_field[x]

    assert result.mu == pytest.approx(-0.243, rel=0.1)
    assert result.sigma == pytest.approx(0.466, rel=0.1)
