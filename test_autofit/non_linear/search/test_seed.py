"""
The search-level ``seed`` (search-extensibility A4b, decision D4;
``autofit.non_linear.search.seed``).

``seed_golden.json`` freezes, per search, what PyAutoFit main 8437a3b76 computed
(``main_*``: default and ``seed=7`` identifiers, output-path identifiers for a model,
the default ``hash_list`` and the default ``search.json``) next to the identifiers this
phase gives an explicit seed (``seed7``, ``seed42``). Identifiers are computed with
``PYAUTO_TEST_MODE`` unset.

The cases, kept separate as the A4 witness asks:

- **default** and **explicit-None** constructions: byte-identical to main for all 15
  (explicit None equals the default everywhere except NSS, as on main).
- **old incomplete output**: a default-constructed search (and a seeded Nautilus/NSS)
  resolves the same output folder as main, so an interrupted fit resumes into it.
- **explicit seed**: Nautilus and NSS unchanged (``"seed"`` is a static field). Every
  other search now hashes a seed it is given; for BlackJAXNUTS and SMC (which took an
  unhashed ``seed`` before) and the four MultiStart searches (likewise) this is the
  accepted re-key of existing explicit-seed output folders. BlackJAXNUTS and SMC read
  ``seed=42`` as their default (adversary review 07, finding 1), so 42 is not re-keyed.
- **old JSON / pickle / output**: a main ``search.json`` loads and round-trips with the
  default identifier for every search, including BlackJAXNUTS and SMC, whose old JSON
  and pickles record ``seed: 42``; the aggregator keeps their output identifiers.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import autofit as af
from autonerves.conf import with_config
from autofit.mapper.identifier import Identifier
from autofit.non_linear.search.seed import (
    backend_generator,
    backend_seed_sequences,
    initializer_rng,
)

from test_autofit.non_linear.search.conformance_roster import searches_under_test

pytestmark = [
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

GOLDEN = json.loads((Path(__file__).parent / "seed_golden.json").read_text())

ROSTER = {entry.name: entry for entry in searches_under_test()}

STATIC_SEED = {"Nautilus", "NSS"}
UNHASHED_DEFAULT_42 = {"BlackJAXNUTS", "SMC"}
HAD_UNHASHED_SEED = UNHASHED_DEFAULT_42 | {
    "MultiStartAdam",
    "MultiStartADABelief",
    "MultiStartLion",
    "MultiStartProdigy",
}


@pytest.fixture(autouse=True)
def no_test_mode(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)


def _cls(name):
    entry = ROSTER[name]
    missing = entry.missing_requirements()
    if missing:
        pytest.skip(f"{name} cannot be constructed without {missing}")
    return entry.resolve()


def _path_identifier(search):
    search.paths.model = af.Model(af.ex.Gaussian)
    return search.paths.identifier


NAMES = sorted(GOLDEN)


def test_golden_covers_the_roster():
    assert set(NAMES) == set(ROSTER)
    assert len(NAMES) == 15


@pytest.mark.parametrize("name", NAMES)
def test_default_and_explicit_none_identifiers_unchanged(name):
    cls = _cls(name)

    assert str(Identifier(cls())) == GOLDEN[name]["main_default"]
    # Equal to the default for every search but NSS, whose static "seed" field
    # defaults to 42 (an explicit None hashes differently there, as on main).
    assert str(Identifier(cls(seed=None))) == GOLDEN[name]["main_none"]


@pytest.mark.parametrize("name", NAMES)
def test_default_hash_list_unchanged(name):
    """`Nautilus().hash_list` (and every other search's) is unchanged."""
    assert Identifier(_cls(name)()).hash_list == GOLDEN[name]["main_hash_list"]


@pytest.mark.parametrize("name", NAMES)
def test_old_incomplete_output_resolves_the_same_folder(name):
    cls = _cls(name)

    assert (
        _path_identifier(cls(name="seed_case", path_prefix="golden"))
        == GOLDEN[name]["main_default_path"]
    )
    if name in STATIC_SEED:
        assert (
            _path_identifier(cls(name="seed_case", path_prefix="golden", seed=7))
            == GOLDEN[name]["main_seed7_path"]
        )


@pytest.mark.parametrize("name", NAMES)
def test_explicit_seed(name):
    cls = _cls(name)
    identifier = str(Identifier(cls(seed=7)))

    assert identifier == GOLDEN[name]["seed7"]
    assert cls(seed=7).seed == 7

    if name in STATIC_SEED:
        assert identifier == GOLDEN[name]["main_seed7"]
    else:
        assert identifier != GOLDEN[name]["main_default"]

    if name in HAD_UNHASHED_SEED:
        # The accepted re-key: main ignored this seed in the identifier.
        assert GOLDEN[name]["main_seed7"] == GOLDEN[name]["main_default"]
        assert identifier != GOLDEN[name]["main_seed7"]


@pytest.mark.parametrize("name", sorted(UNHASHED_DEFAULT_42))
def test_nuts_smc_default_is_an_unhashed_42(name):
    cls = _cls(name)

    assert cls().seed is None
    assert cls().backend_seed == 42
    # Review 07 finding 1: main wrote ``seed: 42`` for every default NUTS/SMC, so 42
    # is the default sentinel; an explicit seed=42 shares the default identity.
    assert cls(seed=42).seed is None
    assert cls(seed=42).backend_seed == 42
    assert str(Identifier(cls(seed=42))) == GOLDEN[name]["main_default"]
    assert af.to_dict(cls(seed=42)) == af.to_dict(cls())


@pytest.mark.parametrize("name", NAMES)
def test_old_search_json_loads(name):
    _cls(name)
    old = GOLDEN[name]["main_search_json"]

    loaded = af.from_dict(json.loads(json.dumps(old)))

    assert af.to_dict(af.from_dict(af.to_dict(loaded))) == af.to_dict(loaded)
    if name in UNHASHED_DEFAULT_42:
        # Review 07 finding 1: main's ``seed: 42`` is the legacy default.
        assert GOLDEN[name]["main_search_json"]["arguments"]["seed"] == 42
        assert loaded.seed is None
        assert loaded.backend_seed == 42
    assert str(Identifier(loaded)) == GOLDEN[name]["main_default"]


def _start_points(search):
    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.UniformPrior(lower_limit=1.0, upper_limit=10.0)
    model.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=20.0)
    analysis = af.ex.Analysis(data=np.ones(20), noise_map=np.ones(20))
    fitness = search.make_fitness(analysis=analysis, model=model)
    _, parameters, _ = search.start_points(model=model, fitness=fitness, n=4, plot=False)
    return np.asarray(parameters)


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_same_initial_points(name):
    cls = _cls(name)

    first = _start_points(cls(seed=11))
    again = _start_points(cls(seed=11))
    other = _start_points(cls(seed=12))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_initializer_never_uses_the_global_random_module(monkeypatch):
    import random

    def forbidden(*args, **kwargs):
        raise AssertionError("the initializer used the global random module")

    monkeypatch.setattr(random, "uniform", forbidden)
    monkeypatch.setattr(random, "random", forbidden)

    _start_points(af.Emcee(seed=1))
    _start_points(af.BFGS(seed=1))
    _start_points(af.DynestyStatic(seed=1))


def test_streams_are_independent_and_stable():
    search = af.Emcee(seed=3)

    zero = initializer_rng(search).random(3)
    one, two = backend_seed_sequences(search, k=2)

    assert np.array_equal(zero, initializer_rng(search).random(3))
    assert not np.array_equal(zero, np.random.default_rng(one).random(3))
    assert not np.array_equal(
        np.random.default_rng(one).random(3), np.random.default_rng(two).random(3)
    )
    assert backend_generator(af.Emcee()) is None


def test_fit_context_hands_the_backend_streams_one_onwards():
    from autofit.non_linear.search.fit_context import FitContext

    search = af.Drawer(seed=5)
    ctx = FitContext(
        search=search,
        model=af.Model(af.ex.Gaussian),
        analysis=None,
        fitness=None,
        pool=None,
        test_mode_level=0,
    )

    (first,) = ctx.rng.spawn(1)
    (expected,) = backend_seed_sequences(search, k=1)

    assert np.array_equal(first.generate_state(4), expected.generate_state(4))


def test_grid_children_get_independent_streams_without_rekeying():
    parent = af.Emcee(seed=9)
    child_0 = parent.copy_with_paths(af.DirectoryPaths(name="c0"))
    child_1 = parent.copy_with_paths(af.DirectoryPaths(name="c1"))
    child_0._seed_spawn_key = (0,)
    child_1._seed_spawn_key = (1,)

    draws = [initializer_rng(s).random(3) for s in (parent, child_0, child_1)]

    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])
    assert str(Identifier(child_0)) == str(Identifier(child_1)) == str(
        Identifier(parent)
    )
    assert "_seed_spawn_key" not in af.to_dict(child_0)["arguments"]


def test_grid_search_sets_a_spawn_key_per_job():
    grid_search = af.SearchGridSearch(
        search=af.m.MockSearch(name="grid_seed"), number_of_steps=2
    )
    model = af.Model(af.ex.Gaussian)
    jobs = grid_search.make_jobs(
        model=model,
        analysis=af.m.MockAnalysis(),
        grid_priors=[model.centre],
    )

    assert [job.search_instance._seed_spawn_key for job in jobs] == [(0,), (1,)]


def test_dynesty_rstate_from_backend_stream():
    assert af.DynestyStatic()._rstate_kwargs() == {}

    a = af.DynestyStatic(seed=4)._rstate_kwargs()["rstate"].random(3)
    b = af.DynestyStatic(seed=4)._rstate_kwargs()["rstate"].random(3)

    assert np.array_equal(a, b)


@with_config("general", "output", "samples_to_csv", value=True)
def test_two_seeded_emcee_fits_write_identical_samples(tmp_path):
    """The A4 witness: two `Emcee(seed=1)` fits on one platform, identical samples.csv."""
    pytest.importorskip("emcee")

    x = np.arange(30.0)
    data = af.ex.Gaussian(centre=15.0, normalization=5.0, sigma=4.0).model_data_from(
        xvalues=x
    )
    model = af.Model(af.ex.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    model.normalization = af.UniformPrior(lower_limit=1.0, upper_limit=10.0)
    model.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=10.0)

    csvs = []
    for run in range(2):
        search = af.Emcee(
            name=f"emcee_seed_{run}",
            path_prefix=str(tmp_path),
            nwalkers=10,
            nsteps=30,
            seed=1,
            number_of_cores=1,
            auto_correlation_settings=af.AutoCorrelationsSettings(
                check_for_convergence=False, check_size=5
            ),
        )
        search.fit(
            model=model,
            analysis=af.ex.Analysis(data=data, noise_map=np.ones(30)),
        )
        csv = search.paths._files_path / "samples.csv"
        if csv.exists():
            csvs.append(csv.read_bytes())
        else:  # the output folder was zipped and removed (`remove_files`)
            with zipfile.ZipFile(search.paths._zip_path) as archive:
                csvs.append(archive.read("files/samples.csv"))

    assert csvs[0] == csvs[1]


# ---------------------------------------------------------------------------
# Adversary review 07.
# ---------------------------------------------------------------------------

# The search identifiers PyAutoFit main 8437a3b76 computed for a default
# BlackJAXNUTS / SMC (whose search.json records ``seed: 42``).
MAIN_DEFAULT_IDENTIFIER = {
    "BlackJAXNUTS": "2a6f5217539f82fab8ea42eda9bf3441",
    "SMC": "3baa494f2ca7b37a8c4b85884cadd1f7",
}


@pytest.mark.parametrize("name", sorted(UNHASHED_DEFAULT_42))
def test_old_nuts_smc_pickle_keeps_the_main_identifier(name):
    """Review 07 finding 1: a main pickle carries ``seed = 42`` in its state."""
    import pickle

    cls = _cls(name)
    search = cls()
    state = search.__getstate__()
    state["seed"] = 42

    loaded = cls.__new__(cls)
    loaded.__setstate__(state)
    loaded = pickle.loads(pickle.dumps(loaded))

    assert loaded.seed is None
    assert loaded.backend_seed == 42
    assert str(Identifier(loaded)) == MAIN_DEFAULT_IDENTIFIER[name]


def _old_output_folder(directory, name, identifier_file=None):
    """An output folder as main wrote it: main's search.json and a model."""
    files = directory / "files"
    files.mkdir(parents=True)
    (files / "search.json").write_text(json.dumps(GOLDEN[name]["main_search_json"]))
    (files / "model.json").write_text(json.dumps(af.to_dict(af.Model(af.ex.Gaussian))))
    if identifier_file is not None:
        (directory / ".identifier").write_text(identifier_file)
    return af.SearchOutput(directory)


@pytest.mark.parametrize("name", sorted(UNHASHED_DEFAULT_42))
def test_old_nuts_smc_output_keeps_the_main_identifier_in_the_aggregator(
    name, tmp_path
):
    """
    Review 07 finding 1: an old default NUTS/SMC output read by the aggregator keeps
    main's search identifier, and ``SearchOutput.id`` (the database scrape's key)
    keeps main's output identifier, with and without a persisted ``.identifier``.
    """
    _cls(name)

    recomputed = _old_output_folder(tmp_path / "recomputed", name)

    assert str(Identifier(recomputed.search)) == MAIN_DEFAULT_IDENTIFIER[name]
    assert recomputed.search.seed is None
    assert recomputed.search.backend_seed == 42
    assert recomputed.id == GOLDEN[name]["main_default_path"]

    paths = af.DirectoryPaths(name="seed_case", path_prefix="golden")
    paths.search = af.from_dict(GOLDEN[name]["main_search_json"])
    paths.model = af.Model(af.ex.Gaussian)
    persisted = _old_output_folder(
        tmp_path / "persisted", name, identifier_file=paths._identifier.description
    )

    assert persisted.id == GOLDEN[name]["main_default_path"]


def test_search_output_id_prefers_the_persisted_identifier(tmp_path):
    """
    Review 07 finding 1: ``SearchOutput.id`` is the identifier the search wrote
    (``.identifier`` holds its description, the md5 of which is the identifier), not a
    recompute from today's search and model, which a later hashing change could move.
    """
    from hashlib import md5

    output = _old_output_folder(
        tmp_path / "out", "Emcee", identifier_file="Emcee\nnwalkers\n50"
    )

    assert output.id == md5("Emcee.nwalkers.50".encode("utf-8")).hexdigest()


def test_search_output_id_recomputes_without_a_persisted_identifier(tmp_path):
    output = _old_output_folder(tmp_path / "out", "Emcee")

    assert output.id == str(
        Identifier([output.search, output.model, output.unique_tag])
    )


# Review 07 finding 3: the backends that take a raw seed.
RAW_SEED_BACKENDS = ["Nautilus", "NSS", "BlackJAXNUTS", "SMC"]


def _children(search):
    child_0 = search.copy_with_paths(af.DirectoryPaths(name="c0"))
    child_1 = search.copy_with_paths(af.DirectoryPaths(name="c1"))
    child_0._seed_spawn_key = (0,)
    child_1._seed_spawn_key = (1,)
    return child_0, child_1


@pytest.mark.parametrize("name", RAW_SEED_BACKENDS)
def test_grid_children_get_distinct_raw_backend_seeds(name):
    cls = _cls(name)
    parent = cls(seed=9)
    child_0, child_1 = _children(parent)

    # Standalone: the backend takes the raw seed, exactly as before.
    assert parent.backend_seed == 9
    assert child_0.backend_seed != child_1.backend_seed
    assert 9 not in (child_0.backend_seed, child_1.backend_seed)
    # Child seeds are reproducible and fit a 32-bit signed JAX PRNGKey.
    assert child_0.backend_seed == _children(cls(seed=9))[0].backend_seed
    assert 0 <= child_0.backend_seed < 2**31
    # The spawn key never reaches the identifier or search.json.
    assert str(Identifier(child_0)) == str(Identifier(child_1)) == str(
        Identifier(parent)
    )
    assert "_seed_spawn_key" not in af.to_dict(child_0)["arguments"]


@pytest.mark.parametrize(
    "name",
    ["MultiStartAdam", "MultiStartADABelief", "MultiStartLion", "MultiStartProdigy"],
)
def test_multi_start_grid_children_get_distinct_streams(name):
    cls = _cls(name)
    parent = cls(seed=9)
    child_0, child_1 = _children(parent)

    def draws(search, stream):
        return np.random.default_rng(search._seed_for(stream)).random(4)

    for stream in (0, 1):
        assert not np.array_equal(draws(child_0, stream), draws(child_1, stream))
        # Standalone: exactly the historical (seed, stream) sequence.
        assert np.array_equal(
            draws(parent, stream),
            np.random.default_rng(np.random.SeedSequence([9, stream])).random(4),
        )
    # Unseeded standalone: the historical bare stream index.
    assert cls()._seed_for(0) == 0
    assert cls()._seed_for(1) == 1


@pytest.mark.parametrize("name", ["Nautilus", "NSS"])
def test_raw_seed_backend_uses_backend_seed(name):
    """The backend call sites read ``backend_seed``, not ``seed``."""
    import inspect

    cls = _cls(name)
    source = inspect.getsource(cls)
    assert "seed=self.backend_seed" in source or "PRNGKey(self.backend_seed)" in source
    assert "PRNGKey(self.seed)" not in source
    assert "seed=self.seed," not in source


# Review 07 finding 2: a subclass that already puts ``rstate`` in ``search_kwargs``.
class _SeededDynestyStatic(af.DynestyStatic):
    def __init__(self, *args, rstate_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rstate_seed = rstate_seed

    @property
    def search_kwargs(self):
        return {
            **super().search_kwargs,
            "rstate": np.random.default_rng(self.rstate_seed),
        }


class _SeededDynestyDynamic(af.DynestyDynamic):
    @property
    def search_kwargs(self):
        return {**super().search_kwargs, "rstate": np.random.default_rng(0)}


def test_dynesty_sampler_kwargs_resolve_rstate_once():
    assert "rstate" not in af.DynestyStatic()._sampler_kwargs()
    seeded = af.DynestyStatic(seed=4)._sampler_kwargs()["rstate"].random(3)
    assert np.array_equal(
        seeded, af.DynestyStatic(seed=4)._rstate_kwargs()["rstate"].random(3)
    )

    # An explicit ``rstate`` in ``search_kwargs`` wins over the search-level seed.
    explicit = _SeededDynestyStatic(seed=4, rstate_seed=0)._sampler_kwargs()
    assert np.array_equal(
        explicit["rstate"].random(3), np.random.default_rng(0).random(3)
    )


@pytest.mark.parametrize("cls", [_SeededDynestyStatic, _SeededDynestyDynamic])
@pytest.mark.parametrize("pooled", [False, True])
def test_dynesty_subclass_with_rstate_constructs_its_sampler(cls, pooled, monkeypatch):
    """The real sampler construction, static and dynamic, pooled and not."""
    pytest.importorskip("dynesty")

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(data=np.ones(20), noise_map=np.ones(20))
    search = cls(seed=3, nlive=10) if cls is _SeededDynestyStatic else cls(seed=3)
    fitness = search.make_fitness(analysis=analysis, model=model)
    monkeypatch.setattr(search, "write_uses_pool", lambda uses_pool: None)

    pool = None
    if pooled:

        class _Pool:
            size = 1

            def loglike(self, x):
                return fitness(x)

            def prior_transform(self, x):
                return x

            def map(self, f, xs):
                return list(map(f, xs))

        pool = _Pool()

    sampler = search.search_internal_from(
        model=model,
        fitness=fitness,
        checkpoint_exists=False,
        pool=pool,
        queue_size=1 if pooled else None,
    )

    assert sampler is not None
