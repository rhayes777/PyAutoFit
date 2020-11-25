from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import NamedTuple, Tuple, Dict, Optional, List

import numpy as np

from autofit.graphical.factor_graphs import Factor
from autofit.graphical.messages.abstract import AbstractMessage
from autofit.graphical.mean_field import \
    MeanField, FactorApproximation, Status
from autofit.graphical.expectation_propagation import \
    EPMeanField, AbstractFactorOptimiser
from autofit.graphical.messages import map_dists
from autofit.graphical.utils import add_arrays


class SamplingResult(NamedTuple):
    samples: Dict[str, np.ndarray]
    det_variables: Dict[str, np.ndarray]
    log_weights: np.ndarray
    log_factor: np.ndarray
    log_measure: np.ndarray
    log_propose: np.ndarray
    n_samples: int

    def __add__(self, other: 'SamplingResult') -> 'SamplingResult':
        return merge_sampling_results(self, other)

    @property
    def weights(self):
        return np.exp(self.log_weights)


def merge_sampling_results(*results: SamplingResult) -> SamplingResult:
    # TODO put check to loop over same variables in results.samples
    n = results[0].n_samples

    samples = {}
    for v, x in results[0].samples.items():
        if len(x) == n:
            samples[v] = np.concatenate([r.samples[v] for r in results])
        elif len(x) == 1:
            # FixedMessages are length 1 so they can't be
            # concatenated
            samples[v] = results[0].samples[v]
        else:
            raise ValueError(f"inconsistent sample lengths for {v}")

    det_variables = {
        v: np.concatenate([r.det_variables[v] for r in results])
        for v in results[0].det_variables}
    log_weights = np.concatenate([r.log_weights for r in results])
    log_factor = np.concatenate([r.log_factor for r in results])
    log_measure = np.concatenate([r.log_measure for r in results])
    log_propose = np.concatenate([r.log_propose for r in results])

    n_samples = sum(r.n_samples for r in results)
    return SamplingResult(
        samples, det_variables, log_weights, log_factor,
        log_measure, log_propose, n_samples)


def effective_sample_size(weights: np.ndarray, axis=None) -> np.ndarray:
    return np.sum(weights, axis=axis) ** 2 / np.square(weights).sum(axis=axis)


class SamplingHistory(NamedTuple):
    n_samples: int = 0
    samples: List[SamplingResult] = list()
    messages: tuple = ()

    def __add__(self, other):
        return SamplingHistory(*(
            getattr(self, f) + getattr(other, f)
            for f in self._fields))


class AbstractSampler(AbstractFactorOptimiser):
    def __init__(self, delta=1., deltas=None, sample_kws=None):
        self.deltas = defaultdict(lambda: delta)
        if deltas:
            self.deltas.update(deltas)

        self.sample_kws = defaultdict(dict)
        if sample_kws:
            self.sample_kws.update(sample_kws)

    @abstractmethod
    def __call__(self, factor_approx: "FactorApproximation",
                 last_samples: Optional[SamplingResult] = None) -> SamplingResult:
        pass

    def optimise(
            self, 
            factor: Factor, 
            model_approx: EPMeanField, 
            status: Status = Status(),
    ) -> Tuple[EPMeanField, Status]:
        delta = self.deltas[factor]
        sample_kws = self.sample_kws[factor]

        factor_approx = model_approx.factor_approximation(factor)
        sample = self(factor_approx, **sample_kws)
        model_dist = project_factor_approx_sample(factor_approx, sample)
        projection, status = factor_approx.project(model_dist, delta=delta)
        return model_approx.project(projection, status=status)

class ImportanceSampler(AbstractSampler):
    def __init__(
            self,
            n_samples: int = 200,
            n_resample: int = 100,
            min_n_eff: int = 100,
            max_samples: int = 1000,
            force_sample: bool = True,
            delta: float = 1.,
            deltas = None, 
            sample_kws = None, 
    ):

        self.params = dict(
            n_samples=n_samples, n_resample=n_resample,
            min_n_eff=min_n_eff, max_samples=max_samples,
            force_sample=force_sample)
        self._history = defaultdict(SamplingHistory)

        super().__init__(
            delta=delta, deltas=deltas, sample_kws=sample_kws)

    def sample(self, factor_approx: "FactorApproximation", **kwargs) -> SamplingResult:
        # Update default params 
        params = {**self.params, **kwargs}
        n_samples = params['n_samples']
        messages = ()

        factor = factor_approx.factor
        cavity_dist = factor_approx.cavity_dist
        deterministic_dist = factor_approx.deterministic_dist
        proposal_dist = factor_approx.model_dist

        samples = {
            v: proposal_dist.get(
                v,
                cavity_dist.get(v)
            ).sample(n_samples=n_samples)
            for v in factor.variables
        }
        fval = factor(samples)
        log_factor = fval + np.zeros(
            (n_samples,) + tuple(1 for _ in range(factor.ndim)))

        sample = self._weight_samples(
            factor, samples, fval.deterministic_values, 
            log_factor, cavity_dist,
            deterministic_dist, proposal_dist, n_samples=n_samples)

        self._history[factor] += SamplingHistory(n_samples, [sample], messages)

        return sample

    def last_samples(self, factor):
        samples = self._history[factor].samples
        if samples:
            return samples[-1]
        return None

    @staticmethod
    def _weight_samples(
            factor: "Factor",
            samples: Dict[str, np.ndarray],
            det_vars: Dict[str, np.ndarray],
            log_factor: np.ndarray,
            cavity_dist: Dict[str, AbstractMessage],
            deterministic_dist: Dict[str, AbstractMessage],
            proposal_dist: Dict[str, AbstractMessage],
            n_samples: int
    ) -> SamplingResult:

        log_measure = 0.
        for res in chain(map_dists(cavity_dist, samples),
                         map_dists(deterministic_dist, det_vars)):
        # for res in map_dists(cavity_dist, {**det_vars, **samples}):
            log_measure = add_arrays(
                log_measure, factor.broadcast_variable(*res))

        log_propose = 0.
        for res in map_dists(proposal_dist, samples):
            log_propose = add_arrays(
                log_propose, factor.broadcast_variable(*res))

        log_weights = log_factor + log_measure - log_propose

        assert np.isfinite(log_weights).all()

        return SamplingResult(
            samples=samples,
            det_variables=det_vars,
            log_weights=log_weights,
            log_factor=log_factor,
            log_measure=log_measure,
            log_propose=log_propose,
            n_samples=n_samples
        )

    def reweight_sample(
            self,
            factor_approx: "FactorApproximation",
            sampling_result: SamplingResult
    ) -> SamplingResult:
        return self._weight_samples(
            factor=factor_approx.factor,
            samples=sampling_result.samples,
            det_vars=sampling_result.det_variables,
            log_factor=sampling_result.log_factor,
            cavity_dist=factor_approx.cavity_dist,
            deterministic_dist=factor_approx.deterministic_dist,
            proposal_dist=factor_approx.model_dist,
            n_samples=sampling_result.n_samples)

    def stop_criterion(self, sample: SamplingResult, **kwargs) -> bool:
        params = {**self.params, **kwargs}
        ess = effective_sample_size(sample.weights, 0).mean()
        n = len(sample.weights)

        return ess > params['min_n_eff'] or n > params['max_samples']

    def __call__(
            self,
            factor_approx: "FactorApproximation",
            **kwargs
    ) -> SamplingResult:
        """
        """
        params = {**self.params, **kwargs}
        samples = None
        if params['force_sample']:
            last_samples = None
        else:
            last_samples = self.last_samples(factor_approx.factor)

        while True:
            if samples is None:
                if last_samples is None:
                    samples = self.sample(factor_approx)
                else:
                    # update weights of the sample for the new 
                    # factor approximation
                    samples = self.reweight_sample(factor_approx, last_samples)

                    # test whether the updated weights satisfy the stopping
                    # criterion
                    if self.stop_criterion(samples):
                        break
                    else:
                        # if not then resample
                        last_samples = None
            else:
                samples = samples + self.sample(factor_approx)
                if self.stop_criterion(samples):
                    break

        return samples


def project_factor_approx_sample(
        factor_approx: FactorApproximation,
        sample: SamplingResult) -> Dict[str, AbstractMessage]:

    #Calculate log_norm
    log_weights = sample.log_weights
    # Need to collapse the weights to match the shapes of the different
    # variables
    variable_log_weights = {
        v: factor_approx.factor.collapse(v, log_weights, agg_func=np.sum)
        for v in factor_approx.cavity_dist}
        
    log_weights = log_weights.sum(
        tuple(range(1, log_weights.ndim)))
    # subtract max log_weight for numerical stability
    log_w_max = np.max(log_weights)
    w = np.exp(log_weights - log_w_max)
    log_norm = np.log(w.mean(0)) + log_w_max

    model_dist = MeanField({
        v: factor_approx.factor_dist[v].project(x, variable_log_weights.get(v))
        for v, x in chain(sample.samples.items(), sample.det_variables.items())},
        log_norm=log_norm)
    return model_dist


def project_model(
        model_approx: EPMeanField,
        factor: Factor,
        sampler: AbstractSampler,
        delta: float = 0.5,
        **kwargs
) -> Tuple[EPMeanField, Status]:
    """
    """
    factor_approx = model_approx.factor_approximation(factor)
    sample = sampler(factor_approx, **kwargs)
    model_dist = project_factor_approx_sample(factor_approx, sample)
    projection, status = factor_approx.project(model_dist, delta=delta)
    return model_approx.project(projection, status=status)
