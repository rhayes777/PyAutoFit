import autofit as af
from autofit.mock.mock import MockClassx4


class MockFitness:
    def figure_of_merit_from_parameters(self, parameters):
        return 1.0


class TestInitializePrior:
    def test__prior__initial_samples_sample_priors(self):

        model = af.PriorModel(MockClassx4)
        model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
        model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
        model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
        model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

        initializer = af.InitializerPrior()

        initial_unit_parameters, initial_parameters, initial_figures_of_merit = initializer.initial_samples_from_model(
            total_points=2, model=model, fitness_function=MockFitness()
        )

        assert 0.0 < initial_unit_parameters[0][0] < 1.0
        assert 0.0 < initial_unit_parameters[1][0] < 1.0
        assert 0.0 < initial_unit_parameters[0][1] < 1.0
        assert 0.0 < initial_unit_parameters[1][1] < 1.0
        assert 0.0 < initial_unit_parameters[0][2] < 1.0
        assert 0.0 < initial_unit_parameters[1][2] < 1.0
        assert 0.0 < initial_unit_parameters[0][3] < 1.0
        assert 0.0 < initial_unit_parameters[1][3] < 1.0

        assert 0.099 < initial_parameters[0][0] < 0.101
        assert 0.099 < initial_parameters[1][0] < 0.101
        assert 0.199 < initial_parameters[0][1] < 0.201
        assert 0.199 < initial_parameters[1][1] < 0.201
        assert 0.299 < initial_parameters[0][2] < 0.301
        assert 0.299 < initial_parameters[1][2] < 0.301
        assert 0.399 < initial_parameters[0][3] < 0.401
        assert 0.399 < initial_parameters[1][3] < 0.401

        assert initial_figures_of_merit == 2 * [1.0]


class TestInitializeBall:
    def test__ball__initial_samples_sample_centre_of_priors(self):

        model = af.PriorModel(MockClassx4)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        model.three = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
        model.four = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)

        initializer = af.InitializerBall(lower_limit=0.4999, upper_limit=0.5001)

        initial_unit_parameters, initial_parameters, initial_figures_of_merit = initializer.initial_samples_from_model(
            total_points=2, model=model, fitness_function=MockFitness()
        )

        assert 0.4999 < initial_unit_parameters[0][0] < 0.5001
        assert 0.4999 < initial_unit_parameters[1][0] < 0.5001
        assert 0.4999 < initial_unit_parameters[0][1] < 0.5001
        assert 0.4999 < initial_unit_parameters[1][1] < 0.5001
        assert 0.4999 < initial_unit_parameters[0][2] < 0.5001
        assert 0.4999 < initial_unit_parameters[1][2] < 0.5001
        assert 0.4999 < initial_unit_parameters[0][3] < 0.5001
        assert 0.4999 < initial_unit_parameters[1][3] < 0.5001

        assert 0.499 < initial_parameters[0][0] < 0.501
        assert 0.499 < initial_parameters[1][0] < 0.501
        assert 0.999 < initial_parameters[0][1] < 1.001
        assert 0.999 < initial_parameters[1][1] < 1.001
        assert 1.499 < initial_parameters[0][2] < 1.501
        assert 1.499 < initial_parameters[1][2] < 1.501
        assert 1.999 < initial_parameters[0][3] < 2.001
        assert 1.999 < initial_parameters[1][3] < 2.001

        initializer = af.InitializerBall(lower_limit=0.7999, upper_limit=0.8001)

        initial_unit_parameters, initial_parameters, initial_figures_of_merit = initializer.initial_samples_from_model(
            total_points=2, model=model, fitness_function=MockFitness()
        )

        assert 0.799 < initial_parameters[0][0] < 0.801
        assert 0.799 < initial_parameters[1][0] < 0.801
        assert 1.599 < initial_parameters[0][1] < 1.601
        assert 1.599 < initial_parameters[1][1] < 1.601
        assert 2.399 < initial_parameters[0][2] < 2.401
        assert 2.399 < initial_parameters[1][2] < 2.401
        assert 3.199 < initial_parameters[0][3] < 3.201
        assert 3.199 < initial_parameters[1][3] < 3.201

        assert initial_figures_of_merit == 2 * [1.0]
