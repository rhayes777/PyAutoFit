"""
A ``NonLinearSearch``'s internal state and its samples: the archive and resume-state
strategies (search-extensibility A3, D5; ``autofit.non_linear.checkpoint``,
``docs/design/checkpointing.md``) and the one samples-conversion path over
``raw_samples_from`` (``autofit.non_linear.samples.adapter``).

``SearchInternalStateMixin`` is a base of ``NonLinearSearch`` only; its methods are the
search's own (subclasses override them as before). It defines no ``__init__`` and no
constructor argument, so serialized searches and identifiers are unaffected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from autonerves import conf

from autofit.non_linear.checkpoint import Checkpointer, DillCheckpointer
from autofit.non_linear.samples.samples import Samples

if TYPE_CHECKING:
    from autofit.mapper.prior_model.abstract import AbstractPriorModel

logger = logging.getLogger(__name__)


class SearchInternalStateMixin:
    checkpointer: Checkpointer = DillCheckpointer()
    """
    The archive strategy: where the backend's final state is stored when a fit
    completes, for ``Result.search_internal`` and ``samples_via_internal_from``.
    """

    resume_state: Optional[Checkpointer] = None
    """
    The file an interrupted run resumes from, for searches that genuinely resume
    (``None`` otherwise). Deleted when the fit completes unless it is the archive's
    own file or declares ``retain_after_completion``.
    """

    samples_cls = Samples
    """
    The ``Samples`` class ``samples_via_internal_from`` builds (``SamplesMCMC`` and
    ``SamplesNest`` on the family bases, ``SamplesSMC`` / ``NSSamples`` on those
    searches).
    """

    @property
    def retains_search_internal(self) -> bool:
        """
        Whether ``search_internal/`` survives the end of a fit: always when the
        archive declares ``retain_after_completion`` (its results cannot be rebuilt
        without it), otherwise when the ``output.search_internal`` config is on.
        Constructing or running a search never changes that config.
        """
        return bool(
            self.checkpointer.retain_after_completion
            or conf.instance["output"]["search_internal"]
        )

    def output_search_internal(self, search_internal):
        """
        Archive ``search_internal`` through the search's ``checkpointer`` and, once a
        fit has completed, discard its resume state (there is nothing left to
        resume).

        ``_fit`` of the searches that archive mid-run (BlackJAX NUTS, SMC) also
        calls this after every chunk; their resume state is ``None``.
        """
        self.checkpointer.finalize(self.paths, search_internal)

        resume_state = self.resume_state
        if (
            resume_state is not None
            and not resume_state.retain_after_completion
            and resume_state.filename != self.checkpointer.filename
        ):
            resume_state.discard(self.paths)

    def finalize_search_internal(self, search_internal):
        """
        Archive ``search_internal`` (``output_search_internal``) when the search retains
        its internal state, otherwise remove the ``search_internal`` folder.

        Run before ``.completed`` is written (``_complete_fit``), so a failure here
        leaves the fit resumable.
        """
        if not self.retains_search_internal:
            self.logger.info("Removing search internal folder.")
            self.paths.remove_search_internal()
        elif search_internal is not None:
            self.output_search_internal(search_internal=search_internal)

    def load_search_internal(self):
        """
        The search's internal state as stored in its output folder, or ``None`` when
        its paths store nothing (``NullPaths``, ``DatabasePaths``).
        """
        return self.paths.load_search_internal()

    def samples_from(self, model: AbstractPriorModel, search_internal=None) -> Samples:
        """
        Loads the samples of a non-linear search from its output files.

        The samples can be loaded from one of two files, which are attempted to be loading in the following order:

        1) Load via the internal results of the non-linear search, which are specified to that search's outputs
           (e.g. the .hdf file output by the MCMC method `emcee`).

        2) Load via the `samples.csv` and `samples_info.json` files of the search, which are outputs that are the
           same for all non-linear searches as they are homogenized by autofit.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        """
        try:
            return self.samples_via_internal_from(
                model=model, search_internal=search_internal
            )
        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning(
                f"The samples of {type(self).__name__} could not be loaded from its "
                f"internal results ({type(e).__name__}: {e}), so they are loaded from "
                f"the samples.csv in the output folder instead."
            )
            return self.paths.samples

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ) -> Samples:
        """
        The samples of the search, converted from its internal state.

        ``search_internal`` is the backend's in-memory state when given; otherwise
        it is loaded from the output folder (``load_search_internal``). The search's
        ``raw_samples_from`` maps it onto a ``RawSamples`` and ``samples_from_raw``
        builds the ``samples_cls`` object (log priors, the log likelihood from a log
        posterior, length checks, ``time`` / ``class_path``), the same way for every
        search.

        Raises
        ------
        NotImplementedError
            When the search does not implement ``raw_samples_from``.
        FileNotFoundError
            When no internal state is stored.
        """
        from autofit.non_linear.samples.adapter import samples_from_raw

        if search_internal is None:
            search_internal = self.load_search_internal()

        raw = self.raw_samples_from(model, search_internal)

        return samples_from_raw(
            model=model,
            raw=raw,
            samples_cls=self.samples_cls,
            time=self.timer.time if self.timer else None,
        )

    def raw_samples_from(self, model: AbstractPriorModel, search_internal):
        """
        Map the backend's internal state onto a
        ``autofit.non_linear.samples.adapter.RawSamples``: its parameter vectors,
        log likelihoods (or log posteriors), weights and ``samples_info`` entries,
        exactly as the backend stores them. Every concrete search implements it.
        """
        raise NotImplementedError
