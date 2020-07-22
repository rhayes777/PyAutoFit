import copy

import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="model_promise")
def make_model_promise(phase):
    return phase.result.model.one.parameter


@pytest.fixture(name="grid_search_promise")
def make_grid_search_promise(phase):
    grid_search_phase = af.as_grid_search(
        af.AbstractPhase
    )(
        phase_name="phase_name",
        phase_tag="phase_tag",
        search=af.MockSearch()
    )
    grid_search_phase.model.one = af.PriorModel(mock.MockComponents, component=mock.MockClassx2)
    return grid_search_phase.result.model.one.parameter


@pytest.fixture(name="instance_promise")
def make_instance_promise(phase):
    return phase.result.instance.one.parameter


@pytest.fixture(name="profile_promise")
def make_profile_promise(phase):
    return phase.result.model.one.component


@pytest.fixture(name="last_model")
def make_last_model():
    return af.last.model.one.parameter


@pytest.fixture(name="last_instance")
def make_last_instance():
    return af.last.instance.one.parameter


class TestHasAttr:
    def test_model(self, phase):
        model = phase.result.model
        assert hasattr(model, "one")
        assert not hasattr(model, "gone")

        components = model.one
        assert hasattr(components, "component")
        assert not hasattr(components, "nada")

    def test_instance(self, phase):
        model = phase.result.instance
        assert hasattr(model, "one")
        assert not hasattr(model, "gone")

        components = model.one
        assert hasattr(components, "component")
        assert not hasattr(components, "nada")


class TestLastPromises:
    def test_indexed_hyper(self, collection):
        result = af.last[0].hyper_result.model.populate(collection)
        assert isinstance(result, af.ModelMapper)
        assert af.last.hyper_result[0].model.populate(collection) is result

    def test_second_indexed_hyper(self, collection):
        result = af.MockResult(model=af.ModelMapper(), instance=af.ModelInstance())
        collection.add("next", result)
        result = af.last[-1].hyper_result.model.populate(collection)
        assert isinstance(result, af.ModelMapper)
        assert af.last.hyper_result[-1].model.populate(collection) is result

    def test_model_absolute(self, collection):
        result = af.last.model_absolute(10).one.parameter.populate(collection)

        assert isinstance(result, af.Prior)

    def test_model_relative(self, collection):
        result = af.last.model_relative(10).one.parameter.populate(collection)

        assert isinstance(result, af.Prior)

    def test_optional(self, collection):
        promise = af.last.model.heart
        with pytest.raises(AttributeError):
            promise.populate(collection)

        promise = af.last.model.optional.heart
        result = promise.populate(collection)
        assert result is None

    def test_model(self, last_model):
        assert last_model.path == ("one", "parameter")
        assert last_model.is_instance is False

    def test_instance(self, last_instance):
        assert last_instance.path == ("one", "parameter")
        assert last_instance.is_instance is True

    def test_recover_model(self, collection, last_model):
        result = last_model.populate(collection)

        assert result is collection[0].model.one.parameter

    def test_recover_instance(self, collection, last_instance):
        result = last_instance.populate(collection)

        assert result is collection[0].instance.one.parameter

    def test_recover_last_model(self, collection, last_model):
        last_results = copy.deepcopy(collection.last)
        collection.add("last_phase", last_results)

        result = last_model.populate(collection)
        assert result is last_results.model.one.parameter
        assert result is not collection[0].model.one.parameter

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

    def test_grid_search_populate(self):
        collection = af.ResultsCollection()
        components = af.PriorModel(mock.MockComponents)
        collection.add(
            "phase one", af.GridSearchResult(
                [
                    af.MockResult(
                        model=af.ModelMapper(component=components)
                    )
                ],
                [[1]],
                [[1]]
            )
        )

        result = af.last.model.component.populate(collection)
        assert result is components

    def test_populate(self):
        collection = af.ResultsCollection()
        components = af.PriorModel(mock.MockComponents)
        model_1 = af.ModelMapper(component=components)

        collection.add("phase one", af.MockResult(model=model_1, instance=None))

        components_2 = af.PriorModel(mock.MockComponents)
        model_2 = af.ModelMapper(component=components_2)

        collection.add("phase two", af.MockResult(model=model_2, instance=None))

        result = af.last.model.component.populate(collection)
        assert result is components_2

        result = af.last[-1].model.component.populate(collection)
        assert result is components

    def test_results_collection_duplicates(self):
        collection = af.ResultsCollection()
        result = af.MockResult(None, None)

        collection.add("name", result)
        collection.add("name", result)

        assert len(list(collection.reversed)) == 1


class TestCase:
    def test_does_not_contribute_to_prior_count(
            self,
            model_promise
    ):
        model = af.ModelMapper(
            argument=model_promise
        )
        assert model.prior_count == 0

    def test_model_promise(self, model_promise, phase):
        assert isinstance(model_promise, af.prior.Promise)
        assert model_promise.path == ("one", "parameter")
        assert model_promise.is_instance is False
        assert model_promise._phase is phase

    def test_grid_search_promise(self, grid_search_promise, phase):
        assert isinstance(grid_search_promise, af.prior.Promise)
        assert grid_search_promise.path == ("one", "parameter")
        assert grid_search_promise.is_instance is False
        assert grid_search_promise._phase is not phase

    def test_optional(self, collection, phase):
        promise = phase.result.model.optional.heart
        result = promise.populate(collection)
        assert result is None

    def test_optional_in_sub(self, collection, phase):
        promise = phase.result.hyper.model.optional.heart
        result = promise.populate(collection)
        assert result is None

    def test_instance_promise(self, instance_promise, phase):
        assert isinstance(instance_promise, af.prior.Promise)
        assert instance_promise.path == ("one", "parameter")
        assert instance_promise.is_instance is True
        assert instance_promise._phase is phase

    def test_non_existent(self, phase):
        with pytest.raises(AttributeError):
            assert phase.result.model.one.bad

        with pytest.raises(AttributeError):
            assert phase.result.instance.one.bad

    def test_recover_model(self, collection, model_promise):
        result = model_promise.populate(collection)

        assert result is collection[0].model.one.parameter

    def test_recover_instance(self, collection, instance_promise):
        result = instance_promise.populate(collection)

        assert result is collection[0].instance.one.parameter

    def test_populate_prior_model_model(self, collection, model_promise):
        new_galaxy = af.PriorModel(mock.MockComponents, parameter=model_promise)

        result = new_galaxy.populate(collection)

        assert result.parameter is collection[0].model.one.parameter

    def test_populate_prior_model_instance(self, collection, instance_promise):
        new_galaxy = af.PriorModel(mock.MockComponents, parameter=instance_promise)

        result = new_galaxy.populate(collection)

        assert result.parameter is collection[0].instance.one.parameter

    def test_kwarg_promise(self, profile_promise, collection):
        components = af.PriorModel(mock.MockComponents, component=profile_promise)
        populated = components.populate(collection)

        assert isinstance(populated.component, af.PriorModel)

        instance = populated.instance_from_prior_medians()

        assert isinstance(instance.kwargs["component"], mock.MockClassx2)

    def test_embedded_results(self, phase, collection):
        hyper_result = phase.result.hyper_result

        assert isinstance(hyper_result, af.prior.PromiseResult)

        model_promise = hyper_result.model
        instance_promise = hyper_result.instance

        assert isinstance(model_promise.hyper_galaxy, af.prior.Promise)
        assert isinstance(instance_promise.hyper_galaxy, af.prior.Promise)

        model = model_promise.populate(collection)
        instance = instance_promise.populate(collection)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert model.hyper_galaxy.cls is mock.HyperGalaxy
        assert isinstance(instance.hyper_galaxy, mock.HyperGalaxy)
