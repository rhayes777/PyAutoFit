"""
The one search-level ``seed`` (search-extensibility phase A4, decision D4).

Every search accepts ``seed`` (``NonLinearSearch.__init__``). It is fanned out with a
``numpy.random.SeedSequence``:

- **stream 0** seeds the initializer (``AbstractInitializer.samples_from_model`` draws
  from a ``numpy.random.Generator``, never from the global ``random`` module);
- **streams 1+** seed the backend: ``FitContext.rng`` has stream 0 spawned already, so
  ``ctx.rng.spawn(k)`` hands a ``run(ctx)`` backend streams ``1..k``; the legacy
  ``_fit`` searches take theirs from ``backend_seed_sequences``;
- **grid-search children** each get an independent child sequence through a spawn key
  (``_seed_spawn_key``), which is neither serialized nor hashed.

The seed enters the search identifier through ``identifier_seed``: the identifier walk
(``autofit.mapper.identifier``) appends it when it is not ``None``, for a search whose
``__identifier_fields__`` do not already name ``"seed"``. Never by adding ``"seed"`` to
the base ``__identifier_fields__``, which would hash the field name for every search.
The legacy table:

=====================  ==========================================================
Search                 Seed rule
=====================  ==========================================================
Nautilus, NSS          ``"seed"`` stays a static identifier field (hashed as
                       before); their ``seed`` argument *is* the base seed.
BlackJAXNUTS, SMC      ``seed=None`` (the default) runs the backend with the
                       historical ``42`` and is not hashed; any other seed is
                       hashed. ``seed=42`` *is* the default: main recorded
                       ``seed: 42`` in every default NUTS/SMC ``search.json`` and
                       pickle, so a 42 is read back as ``None`` (``legacy_default_seed``)
                       and an old output keeps its identifier.
every other search     hashed only when not ``None``.
=====================  ==========================================================

What a seed promises: the same seed gives the same initial points for every search.
Bit-identical samples hold only within one backend and platform where that has been
demonstrated (e.g. Emcee on one machine); MCMC archive searches resume statistically,
not exactly. Backends that take the seed directly (Nautilus's ``Sampler(seed=)``, the
JAX ``PRNGKey`` of NSS, NUTS and SMC, MultiStart's own ``(seed, stream)`` sequences)
keep doing so, so an existing seeded run of those searches is reproduced exactly.
"""

from typing import List, Optional

import numpy as np

# The seed main hard-coded as the BlackJAXNUTS / SMC default (recorded in their
# ``search.json`` and pickles as ``seed: 42``).
LEGACY_DEFAULT_SEED = 42


def legacy_default_seed(seed: Optional[int]) -> Optional[int]:
    """
    The BlackJAXNUTS / SMC seed with main's default ``42`` read as ``None``.

    main ran these backends with an unhashed ``seed=42`` by default and wrote ``seed: 42``
    into every default ``search.json`` and pickle, whether the 42 was the default or
    passed explicitly. Reading a 42 as the unhashed default keeps every such output's
    identifier (and ``SearchOutput.id``) what main computed; the backend still runs
    with 42, so sampling is bit-identical. The cost is that an explicit ``seed=42``
    shares the default identity.
    """
    if seed is not None and not isinstance(seed, bool) and seed == LEGACY_DEFAULT_SEED:
        return None
    return seed


def seed_sequence(search) -> np.random.SeedSequence:
    """
    A fresh ``SeedSequence`` of the search's seed (and grid-child spawn key).

    ``SeedSequence(None)`` draws fresh OS entropy, so an unseeded search is not
    reproducible, exactly as before; a fresh sequence per call keeps the stream indices
    stable however often it is asked.
    """
    return np.random.SeedSequence(
        getattr(search, "seed", None),
        spawn_key=tuple(getattr(search, "_seed_spawn_key", ()) or ()),
    )


def initializer_rng(search) -> np.random.Generator:
    """
    The generator the initializer draws the search's start points from: stream 0.
    """
    return np.random.default_rng(seed_sequence(search).spawn(1)[0])


def backend_seed_sequences(search, k: int = 1) -> List[np.random.SeedSequence]:
    """
    The backend's ``k`` streams: streams ``1..k`` of the search's seed sequence.
    """
    return seed_sequence(search).spawn(k + 1)[1:]


def backend_random_state(search) -> Optional[np.random.RandomState]:
    """
    A legacy ``numpy.random.RandomState`` on backend stream 1 when the search has a
    seed, for backends built on ``RandomState`` (``emcee``); ``None`` when unseeded, so
    the backend keeps its own default.
    """
    if getattr(search, "seed", None) is None:
        return None
    return np.random.RandomState(np.random.MT19937(backend_seed_sequences(search)[0]))


def backend_generator(search) -> Optional[np.random.Generator]:
    """
    A ``numpy.random.Generator`` on backend stream 1 when the search has a seed (e.g.
    dynesty's ``rstate``); ``None`` when unseeded.
    """
    if getattr(search, "seed", None) is None:
        return None
    return np.random.default_rng(backend_seed_sequences(search)[0])
