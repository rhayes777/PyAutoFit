import copy

import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="phase")
def make_phase():
    phase = af.AbstractPhase(phase_name="phase name")
    phase.model.one = af.PriorModel(mock.Galaxy, light=mock.EllipticalLP)
    return phase


@pytest.fixture(name="model_promise")
def make_model_promise(phase):
    return phase.result.model.one.redshift


@pytest.fixture(name="instance_promise")
def make_instance_promise(phase):
    return phase.result.instance.one.redshift


@pytest.fixture(name="profile_promise")
def make_profile_promise(phase):
    return phase.result.model.one.light


@pytest.fixture(name="collection")
def make_collection():
    collection = af.ResultsCollection()
    model = af.ModelMapper()
    model.one = af.PriorModel(mock.Galaxy, light=mock.EllipticalLP)
    instance = af.ModelInstance()
    instance.one = mock.Galaxy(light=mock.EllipticalLP())

    result = mock.Result(model=model, instance=instance)

    model = af.ModelMapper()
    instance = af.ModelInstance()

    model.hyper_galaxy = mock.HyperGalaxy
    instance.hyper_galaxy = mock.HyperGalaxy()

    hyper_result = mock.Result(model=model, instance=instance)

    result.hyper_result = hyper_result

    collection.add("phase name", result)

    return collection


@pytest.fixture(name="last_model")
def make_last_model():
    return af.last.model.one.redshift


@pytest.fixture(name="last_instance")
def make_last_instance():
    return af.last.instance.one.redshift


class TestLastPromises:
    def test_model_absolute(self, collection):
        result = af.last.model_absolute(10).one.redshift.populate(collection)

        assert isinstance(result, af.Prior)

    def test_model_relative(self, collection):
        result = af.last.model_relative(10).one.redshift.populate(collection)

        assert isinstance(result, af.Prior)

    def test_optional(self, collection):
        promise = af.last.model.heart
        with pytest.raises(AttributeError):
            promise.populate(collection)

        promise = af.last.model.optional.heart
        result = promise.populate(collection)
        assert result is None

    def test_model(self, last_model):
        assert last_model.path == ("one", "redshift")
        assert last_model.is_instance is False

    def test_instance(self, last_instance):
        assert last_instance.path == ("one", "redshift")
        assert last_instance.is_instance is True

    def test_recover_model(self, collection, last_model):
        result = last_model.populate(collection)

        assert result is collection[0].model.one.redshift

    def test_recover_instance(self, collection, last_instance):
        result = last_instance.populate(collection)

        assert result is collection[0].instance.one.redshift

    def test_recover_last_model(self, collection, last_model):
        last_results = copy.deepcopy(collection.last)
        collection.add("last_phase", last_results)

        result = last_model.populate(collection)
        assert result is last_results.model.one.redshift
        assert result is not collection[0].model.one.redshift

    def test_embedded_results(self, collection):
        hyper_result = af.last.hyper_result

        model_promise = hyper_result.model
        instance_promise = hyper_result.instance

        model = model_promise.populate(collection)
        instance = instance_promise.populate(collection)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert model.hyper_galaxy.cls is mock.HyperGalaxy
        assert isinstance(instance.hyper_galaxy, mock.HyperGalaxy)

    def test_raises(self, collection):
        bad_promise = af.last.model.a.bad.path
        with pytest.raises(AttributeError):
            bad_promise.populate(collection)


class TestIndexLast:
    def test_index(self):
        assert af.last._index == 0
        assert af.last[-1]._index == -1
        with pytest.raises(IndexError):
            _ = af.last[1]

    def test_populate(self):
        collection = af.ResultsCollection()
        galaxy_model_1 = af.PriorModel(mock.Galaxy)
        model_1 = af.ModelMapper(galaxy=galaxy_model_1)

        collection.add("phase one", mock.Result(model=model_1, instance=None))

        galaxy_model_2 = af.PriorModel(mock.Galaxy)
        model_2 = af.ModelMapper(galaxy=galaxy_model_2)

        collection.add("phase two", mock.Result(model=model_2, instance=None))

        result = af.last.model.galaxy.populate(collection)
        assert result is galaxy_model_2

        result = af.last[-1].model.galaxy.populate(collection)
        assert result is galaxy_model_1


class TestCase:
    def test_model_promise(self, model_promise, phase):
        assert isinstance(model_promise, af.Promise)
        assert model_promise.path == ("one", "redshift")
        assert model_promise.is_instance is False
        assert model_promise.phase is phase

    def test_optional(self, collection, phase):
        promise = phase.result.model.optional.heart
        result = promise.populate(collection)
        assert result is None

    def test_instance_promise(self, instance_promise, phase):
        assert isinstance(instance_promise, af.Promise)
        assert instance_promise.path == ("one", "redshift")
        assert instance_promise.is_instance is True
        assert instance_promise.phase is phase

    def test_non_existent(self, phase):
        with pytest.raises(AttributeError):
            assert phase.result.model.one.bad

        with pytest.raises(AttributeError):
            assert phase.result.instance.one.bad

    def test_recover_model(self, collection, model_promise):
        result = model_promise.populate(collection)

        assert result is collection[0].model.one.redshift

    def test_recover_instance(self, collection, instance_promise):
        result = instance_promise.populate(collection)

        assert result is collection[0].instance.one.redshift

    def test_populate_prior_model_model(self, collection, model_promise):
        new_galaxy = af.PriorModel(mock.Galaxy, redshift=model_promise)

        result = new_galaxy.populate(collection)

        assert result.redshift is collection[0].model.one.redshift

    def test_populate_prior_model_instance(self, collection, instance_promise):
        new_galaxy = af.PriorModel(mock.Galaxy, redshift=instance_promise)

        result = new_galaxy.populate(collection)

        assert result.redshift is collection[0].instance.one.redshift

    def test_kwarg_promise(self, profile_promise, collection):
        galaxy = af.PriorModel(mock.Galaxy, light=profile_promise)
        populated = galaxy.populate(collection)

        assert isinstance(populated.light, af.PriorModel)

        instance = populated.instance_from_prior_medians()

        assert isinstance(instance.kwargs["light"], mock.EllipticalLP)

    def test_embedded_results(self, phase, collection):
        hyper_result = phase.result.hyper_result

        assert isinstance(hyper_result, af.PromiseResult)

        model_promise = hyper_result.model
        instance_promise = hyper_result.instance

        print(model_promise.path)

        assert isinstance(model_promise.hyper_galaxy, af.Promise)
        assert isinstance(instance_promise.hyper_galaxy, af.Promise)

        model = model_promise.populate(collection)
        instance = instance_promise.populate(collection)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert model.hyper_galaxy.cls is mock.HyperGalaxy
        assert isinstance(instance.hyper_galaxy, mock.HyperGalaxy)
