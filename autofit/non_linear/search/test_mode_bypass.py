"""
Test mode for non-linear searches (``PYAUTO_TEST_MODE``).

- Level 1 runs the real backend with a reduced budget. The reduction is the search's
  declared ``test_mode_budget`` (``autofit.non_linear.search.capabilities``), applied
  once after construction by ``apply_test_mode`` (the post-init hook of
  ``NonLinearSearch``). A reduced fit whose final sample is a rejected point is
  repaired by ``TestModeBypass.samples_after_rejected_fit``.
- Levels 2 and 3 bypass the backend entirely: ``TestModeBypass.run`` writes fake but
  complete outputs, calling the likelihood once (level 2) or never (level 3).

Both the bypass and the real fit end in the search's one ``finalize_result``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import numpy as np

from autofit import exc
from autofit.non_linear.test_mode import is_test_mode, test_mode_level, test_mode_samples

if TYPE_CHECKING:
    from autofit.mapper.prior_model.abstract import AbstractPriorModel
    from autofit.non_linear.analysis import Analysis
    from autofit.non_linear.samples.samples import Samples

logger = logging.getLogger(__name__)


# A reduced test-mode fit needs a valid representative instance for result
# construction and search chaining.  Keep fallback sampling bounded so an
# impossible model fails clearly instead of hanging a smoke-test worker.
TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS = 100


def apply_test_mode(search):
    """
    Apply the search's declared ``test_mode_budget`` (``PYAUTO_TEST_MODE=1``).

    Every key names an attribute of the search, dotted for an attribute of one of its
    members (``"convergence.window"``), and is set to the budget value. A search whose
    reduction is not a plain assignment (``BlackJAXNUTS`` and ``SMC`` cap a user's
    smaller value rather than raise it) overrides ``apply_test_mode`` instead.
    """
    budget = type(search).test_mode_budget

    if not budget:
        return

    logger.warning(
        f"TEST MODE 1 (reduced iterations): {type(search).__name__} will run with "
        + ", ".join(f"{key}={value}" for key, value in budget.items())
        + " for faster completion."
    )

    for key, value in budget.items():
        *owners, attribute = key.split(".")
        target = search
        for owner in owners:
            target = getattr(target, owner)
        setattr(target, attribute, value)


class TestModeBypass:
    """
    The test-mode collaborator of one search: the level 2/3 bypass and the level 1
    repair of a rejected final sample. It reads only the search's paths, its
    ``_test_mode_samples_info`` hook and its ``finalize_result``.
    """

    # A library class whose name starts with ``Test``: never a pytest test class.
    __test__ = False

    def __init__(self, search):
        self.search = search

    def run(self, model: AbstractPriorModel, analysis: Analysis, call_likelihood: bool = True):
        """
        Bypass the sampler entirely in test mode (levels 2 and 3).

        Generates fake samples and writes all expected output files so that
        downstream code sees a complete result folder.

        Parameters
        ----------
        model
            The model being fitted.
        analysis
            The analysis object with the log likelihood function.
        call_likelihood
            If True (mode 2), call the likelihood function once to verify it
            works. If False (mode 3), skip the likelihood call entirely.
        """
        from autofit.non_linear.samples.pdf import SamplesPDF

        if test_mode_level() == 2:
            logger.warning(
                "TEST MODE 2 (bypass + likelihood): Skipping sampler, "
                "calling likelihood function once to verify it works."
            )
        else:
            logger.warning(
                "TEST MODE 3 (full bypass): Skipping sampler and likelihood "
                "entirely for maximum speed. No likelihood verification."
            )

        model.freeze()

        # The bypass has no sampler, so it chooses its own evaluation point and
        # must choose one the model accepts. A model whose components share
        # priors and carry an ordering assertion ties exactly at the prior
        # medians, and `check_assertions` rejects that tie with `FitException`.
        # The vector chosen here is also the one written into the fake samples
        # below, so picking a valid one is what keeps
        # `result.max_log_likelihood_instance` reconstructible — which is why
        # mode 3 needs this too, not just mode 2's likelihood call
        # (PyAutoFit #1519).
        parameter_vector, instance = self.valid_parameter_vector(
            model=model,
            failure_prefix=(
                "TEST MODE could not find a parameter vector satisfying the "
                "model's assertions"
            ),
        )

        log_likelihood = -1.0e99
        if call_likelihood:
            try:
                log_likelihood = float(analysis.log_likelihood_function(instance))
            except exc.FitException as e:
                # A `FitException` means this particular instance is pathological
                # (e.g. a non-positive-definite inversion, or a degenerate mesh
                # that yields NaN vertices). In a real search the sampler absorbs
                # this by resampling; test mode has no sampler, so a single
                # unlucky verification eval must not hard-fail the run. Keep the
                # `-1.0e99` sentinel — the same effect a resample-to-reject has —
                # and log the cause so a genuinely broken likelihood stays visible.
                # Only `FitException` is caught: real code errors still propagate.
                logger.warning(
                    "TEST MODE 2: likelihood verification raised FitException "
                    f"({e.__cause__ or e!r}); treating as a resample-rejected "
                    "instance and continuing with the sentinel log likelihood."
                )

        sample_list = self.build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=log_likelihood,
        )

        # Stub log_evidence in samples_info so downstream arithmetic
        # (grid search log_evidences, subhalo Bayesian model comparison,
        # scrape aggregator assertions) doesn't crash on None. SamplesPDF
        # reads log_evidence from samples_info.
        samples_info = {
            "total_iterations": 1,
            "time": 0.0,
            "log_evidence": log_likelihood,
        }
        samples_info.update(self.search._test_mode_samples_info())
        samples = SamplesPDF(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

        samples_summary = samples.summary()

        # Persist samples + summary to disk so downstream code that reads
        # from the output folder (database scrape, paths.load_samples_summary)
        # sees a complete result. NullPaths and DatabasePaths both handle these
        # calls safely.
        self.search.paths.save_samples_summary(samples_summary=samples_summary)
        self.search.paths.save_samples(samples=samples)

        # The real fit's ending: the analysis' own output hooks run (a script that
        # writes a file in save_results and reads it back after the fit sees a
        # complete folder) and the fit is marked complete, so a rerun loads it via
        # `result_via_completed_fit` rather than re-bypassing the whole pipeline.
        return self.search.finalize_result(
            samples=samples,
            model=model,
            analysis=analysis,
            samples_summary=samples_summary,
        )

    def valid_parameter_vector(
        self,
        model: AbstractPriorModel,
        failure_prefix: str,
        validate: Optional[Callable[[List[float]], Any]] = None,
    ) -> Tuple[List[float], Any]:
        """The first deterministically-drawn parameter vector the model accepts.

        Test mode has no sampler, so it evaluates the model at a point it picks
        itself — the prior medians.  A model whose components share priors and
        carry an ordering assertion (``trap_0.release_timescale <
        trap_1.release_timescale``, the standard idiom for breaking exchange
        degeneracy) ties *exactly* there, so ``check_assertions`` rejects the
        medians with :class:`FitException`.  A production search absorbs that by
        resampling, and test mode must do the same rather than hard-fail on an
        artifact of its own choice of point.

        Try the prior medians first, then a deterministic sequence of prior
        draws, returning the first candidate ``validate`` accepts.  The fixed
        seed keeps smoke runs reproducible without touching the application's
        global random state.

        Parameters
        ----------
        model
            The model the vector must be valid for.
        failure_prefix
            Opening of the :class:`FitException` message raised once the attempt
            budget is spent; the attempt count is appended to it.
        validate
            Called with each candidate vector.  Raising `FitException` rejects
            that candidate and moves on to the next draw; the return value is
            handed back alongside the accepted vector, so a caller that must
            build something in order to validate it does not build it twice.
            Defaults to instantiating the vector.

        Returns
        -------
        The accepted parameter vector, and whatever ``validate`` returned for it.

        Raises
        ------
        FitException
            If no candidate is accepted within
            ``TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS`` attempts, chained to the
            final rejection.
        """
        if validate is None:

            def validate(candidate: List[float]):
                return model.instance_from_vector(vector=candidate)

        rng = np.random.default_rng(seed=0)
        last_error = None

        for attempt in range(TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS):
            unit_vector = (
                [0.5] * model.prior_count
                if attempt == 0
                else rng.random(model.prior_count).tolist()
            )

            try:
                parameter_vector = [
                    float(value)
                    for value in model.vector_from_unit_vector(
                        unit_vector=unit_vector,
                    )
                ]
                validated = validate(parameter_vector)
            except exc.FitException as candidate_error:
                last_error = candidate_error
                continue

            if attempt > 0:
                logger.warning(
                    "TEST MODE: the prior medians are not a valid model "
                    f"instance ({last_error.__cause__ or last_error!r}); using "
                    f"deterministic prior draw {attempt} instead. A model whose "
                    "components share priors and carry an ordering assertion "
                    "ties at its medians, which a real search resamples past."
                )

            return parameter_vector, validated

        raise exc.FitException(
            f"{failure_prefix} after "
            f"{TEST_MODE_REPRESENTATIVE_MAX_ATTEMPTS} attempts."
        ) from last_error

    def repair_reduced_samples(self, samples: Samples):
        """
        ``PYAUTO_TEST_MODE=1`` only: when the reduced fit's best sample is a rejected
        point, replace the samples with valid representative ones (and save them).

        Returns the (possibly replaced) samples and their summary.
        """
        samples_summary = samples.summary()

        try:
            samples_summary.max_log_likelihood()
        except (exc.FitException, exc.SamplesException) as error:
            samples = self.samples_after_rejected_fit(samples=samples, error=error)
            samples_summary = samples.summary()
            self.search.paths.save_samples_summary(samples_summary=samples_summary)
            self.search.paths.save_samples(samples=samples)

        return samples, samples_summary

    def samples_after_rejected_fit(self, samples: Samples, error: Exception) -> Samples:
        """Build valid representative samples after a mode-1 rejected result.

        ``Fitness`` maps :class:`FitException` to the sampler's rejection
        sentinel.  A production search naturally moves on to another point,
        but ``PYAUTO_TEST_MODE=1`` may stop after that first evaluation.  Its
        posterior can therefore contain only the rejected point, which cannot
        be reconstructed while finalizing the result.

        Try the prior medians first, then a deterministic sequence of prior
        draws.  Every synthetic sample is validated before it is returned so
        result construction and downstream chaining cannot reconstruct another
        rejected point.  The fixed seed keeps smoke tests reproducible without
        changing the application's global random state.
        """
        logger.warning(
            "TEST MODE 1: the reduced search's final sample raised "
            f"FitException ({error.__cause__ or error!r}); replacing it with "
            "a valid representative sample for result construction."
        )

        model = samples.model

        def validate(parameter_vector: List[float]):
            """Every synthetic sample, not just the first, must reconstruct."""
            sample_list = self.build_fake_samples(
                model=model,
                parameter_vector=parameter_vector,
                log_likelihood=-1.0e99,
            )

            for sample in sample_list:
                model.instance_from_vector(
                    vector=sample.parameter_lists_for_model(model)
                )

            return sample_list

        _, sample_list = self.valid_parameter_vector(
            model=model,
            failure_prefix=(
                "TEST MODE 1 could not construct a valid representative result"
            ),
            validate=validate,
        )

        samples_info = {
            **(samples.samples_info or {}),
            "total_iterations": 1,
            "time": 0.0,
            "log_evidence": -1.0e99,
        }
        samples_info.update(self.search._test_mode_samples_info())

        return samples.from_list_info_and_model(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )

    @staticmethod
    def build_fake_samples(model, parameter_vector, log_likelihood):
        """
        Build a list of fake Sample objects for test mode bypass.

        Creates a deterministic sample set: the "best" at the prior median
        and additional slightly perturbed parameters with worse likelihoods.
        The default of four samples keeps bypass mode cheap while allowing
        downstream structural checks to exercise multi-batch sample handling.

        ``PYAUTO_TEST_MODE_SAMPLES=N`` raises the sample count so the
        bypass run's ``samples.csv`` row count and byte size match a
        production sampler stage (N ~ 10k-100k), keeping resume/load
        timings measured against the output honest. The N > 4 samples are
        synthesized vectorized (numpy) then materialised through the same
        ``Sample.from_lists`` path a real sampler run uses, so structure
        and cost are representative by construction. The best (first)
        sample is the unperturbed prior median in both branches.
        """
        from autofit.non_linear.samples.sample import Sample

        total_samples = test_mode_samples()

        if total_samples == 4:
            parameter_lists = [parameter_vector]
            for scale in (1.001, 0.999, 1.002):
                parameter_lists.append(
                    [p * scale if p != 0.0 else scale - 1.0 for p in parameter_vector]
                )

            return Sample.from_lists(
                model=model,
                parameter_lists=parameter_lists,
                log_likelihood_list=[
                    log_likelihood - offset for offset in range(len(parameter_lists))
                ],
                log_prior_list=[0.0] * len(parameter_lists),
                weight_list=[1.0, 0.5, 0.25, 0.125],
            )

        rng = np.random.default_rng(0)
        base = np.asarray(parameter_vector, dtype=float)
        scatter = 1.0e-3 * rng.standard_normal((total_samples, base.shape[0]))
        parameters = np.where(base == 0.0, scatter, base * (1.0 + scatter))
        parameters[0] = base

        # Weights decay over ~N/10 samples so the effective sample size stays
        # a healthy fraction of N and the smallest weight, ~(10/N)e^-10, sits
        # above the output.yaml samples_weight_threshold of 1e-10 for N <= 1e5.
        weights = np.exp(
            -np.arange(total_samples, dtype=float) / (total_samples / 10.0)
        )

        return Sample.from_lists(
            model=model,
            parameter_lists=parameters.tolist(),
            log_likelihood_list=(
                log_likelihood - np.arange(total_samples, dtype=float)
            ).tolist(),
            log_prior_list=[0.0] * total_samples,
            weight_list=(weights / weights.sum()).tolist(),
        )


class TestModeMixin:
    """
    The test-mode hooks of ``NonLinearSearch`` (a base of it only; no ``__init__``,
    no constructor argument).
    """

    def _post_init(self):
        """
        The post-init hook, run once after construction (``_SearchMeta.__call__``,
        after the outermost ``__init__`` returns, so a subclass's own attributes are
        set first): under ``PYAUTO_TEST_MODE`` it applies the search's test-mode
        reduction. This replaces the ``if is_test_mode(): self.apply_test_mode()``
        tail every search used to repeat at the end of its ``__init__``.
        """
        if is_test_mode():
            self.apply_test_mode()

    def apply_test_mode(self):
        """
        Reduce the search's budget for test mode: the generic reduction sets each
        attribute of the declared ``test_mode_budget`` (``apply_test_mode`` of this
        module). Called once, by the post-init hook; override only for a reduction that
        is not a plain assignment.
        """
        apply_test_mode(self)

    def _test_mode_samples_info(self) -> dict:
        """
        Sampler-specific keys to merge into ``samples_info`` when the
        sampler is bypassed via ``PYAUTO_TEST_MODE=2`` or ``=3``.

        **Opt-in, not a per-sampler obligation.** Most searches do not
        override this, and that is correct — no library code reads these
        diagnostic keys under bypass. The properties that read them
        (``SamplesMCMC.total_steps``, ``SamplesNest.total_samples``, …)
        live on ``Samples`` subclasses the bypass never constructs;
        ``TestModeBypass.run`` always builds a ``SamplesPDF``. The only
        consumers are workspace scripts reading ``samples_info[...]``
        directly, so override this only when such a script exists.

        Which fix applies depends on what that script does with the keys:

        - It **prints** them (a tutorial, whose point is the prose, so it
          may legitimately run bypassed) → override here, returning NaN/0
          placeholders. The bypass did not sample; honest empties.
        - It **asserts** on them (a test) → the script must not run
          bypassed at all. Give it an ``ENV: real_search`` declaration
          instead. Do **not** add placeholders for an asserting reader:
          the assert then silently passes on a stub value, which is worse
          than the ``KeyError`` it replaced.

        Both live cases follow that split. ``BlackJAXNUTS`` overrides this
        because ``autofit_workspace/scripts/searches/mcmc.py`` prints
        ``ess_min`` / ``n_divergent``, carries no ``__Env__`` declaration
        and is in that workspace's ``smoke_tests.txt``, so it runs
        bypassed on every PR (PyAutoFit #1260).
        ``AbstractMultiStartGradient`` deliberately does not: its only
        reader asserts on ``total_steps``, so that script declares
        ``ENV: real_search jax`` instead (autofit_workspace_test #83).
        """
        return {}

    @staticmethod
    def _build_fake_samples(model, parameter_vector, log_likelihood):
        """
        The fake samples of the test-mode bypass (``TestModeBypass.build_fake_samples``).
        """
        return TestModeBypass.build_fake_samples(model, parameter_vector, log_likelihood)
