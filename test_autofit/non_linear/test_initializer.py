import autofit as af


class MockFitness:
    def figure_of_merit_from(self, parameter_list):
        return 1.0


class TestInitializePrior:
    def test__prior__samples_sample_priors(self):

        model = af.PriorModel(af.m.MockClassx4)
        model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
        model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
        model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
        model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

        initializer = af.InitializerPrior()

        unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
            total_points=2, model=model, fitness_function=MockFitness()
        )

        assert 0.0 < unit_parameter_lists[0][0] < 1.0
        assert 0.0 < unit_parameter_lists[1][0] < 1.0
        assert 0.0 < unit_parameter_lists[0][1] < 1.0
        assert 0.0 < unit_parameter_lists[1][1] < 1.0
        assert 0.0 < unit_parameter_lists[0][2] < 1.0
        assert 0.0 < unit_parameter_lists[1][2] < 1.0
        assert 0.0 < unit_parameter_lists[0][3] < 1.0
        assert 0.0 < unit_parameter_lists[1][3] < 1.0

        assert 0.099 < parameter_lists[0][0] < 0.101
        assert 0.099 < parameter_lists[1][0] < 0.101
        assert 0.199 < parameter_lists[0][1] < 0.201
        assert 0.199 < parameter_lists[1][1] < 0.201
        assert 0.299 < parameter_lists[0][2] < 0.301
        assert 0.299 < parameter_lists[1][2] < 0.301
        assert 0.399 < parameter_lists[0][3] < 0.401
        assert 0.399 < parameter_lists[1][3] < 0.401

        assert figure_of_merit_list == [1.0, 1.0]

    def test__samples_in_test_model(self):

        model = af.PriorModel(af.m.MockClassx4)
        model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
        model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
        model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
        model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

        initializer = af.InitializerPrior()

        unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_in_test_mode(
            total_points=2, model=model,
        )

        assert 0.0 < unit_parameter_lists[0][0] < 1.0
        assert 0.0 < unit_parameter_lists[1][0] < 1.0
        assert 0.0 < unit_parameter_lists[0][1] < 1.0
        assert 0.0 < unit_parameter_lists[1][1] < 1.0
        assert 0.0 < unit_parameter_lists[0][2] < 1.0
        assert 0.0 < unit_parameter_lists[1][2] < 1.0
        assert 0.0 < unit_parameter_lists[0][3] < 1.0
        assert 0.0 < unit_parameter_lists[1][3] < 1.0

        assert 0.099 < parameter_lists[0][0] < 0.101
        assert 0.099 < parameter_lists[1][0] < 0.101
        assert 0.199 < parameter_lists[0][1] < 0.201
        assert 0.199 < parameter_lists[1][1] < 0.201
        assert 0.299 < parameter_lists[0][2] < 0.301
        assert 0.299 < parameter_lists[1][2] < 0.301
        assert 0.399 < parameter_lists[0][3] < 0.401
        assert 0.399 < parameter_lists[1][3] < 0.401

        assert figure_of_merit_list == [-1.0e99, -1.0e99]


class TestInitializeBall:
    def test__ball__samples_sample_centre_of_priors(self):

        model = af.PriorModel(af.m.MockClassx4)
        model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        model.two = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        model.three = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
        model.four = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)

        initializer = af.InitializerBall(lower_limit=0.4999, upper_limit=0.5001)

        unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
            total_points=2, model=model, fitness_function=MockFitness()
        )

        assert 0.4999 < unit_parameter_lists[0][0] < 0.5001
        assert 0.4999 < unit_parameter_lists[1][0] < 0.5001
        assert 0.4999 < unit_parameter_lists[0][1] < 0.5001
        assert 0.4999 < unit_parameter_lists[1][1] < 0.5001
        assert 0.4999 < unit_parameter_lists[0][2] < 0.5001
        assert 0.4999 < unit_parameter_lists[1][2] < 0.5001
        assert 0.4999 < unit_parameter_lists[0][3] < 0.5001
        assert 0.4999 < unit_parameter_lists[1][3] < 0.5001

        assert 0.499 < parameter_lists[0][0] < 0.501
        assert 0.499 < parameter_lists[1][0] < 0.501
        assert 0.999 < parameter_lists[0][1] < 1.001
        assert 0.999 < parameter_lists[1][1] < 1.001
        assert 1.499 < parameter_lists[0][2] < 1.501
        assert 1.499 < parameter_lists[1][2] < 1.501
        assert 1.999 < parameter_lists[0][3] < 2.001
        assert 1.999 < parameter_lists[1][3] < 2.001

        initializer = af.InitializerBall(lower_limit=0.7999, upper_limit=0.8001)

        unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
            total_points=2, model=model, fitness_function=MockFitness()
        )

        assert 0.799 < parameter_lists[0][0] < 0.801
        assert 0.799 < parameter_lists[1][0] < 0.801
        assert 1.599 < parameter_lists[0][1] < 1.601
        assert 1.599 < parameter_lists[1][1] < 1.601
        assert 2.399 < parameter_lists[0][2] < 2.401
        assert 2.399 < parameter_lists[1][2] < 2.401
        assert 3.199 < parameter_lists[0][3] < 3.201
        assert 3.199 < parameter_lists[1][3] < 3.201

        assert figure_of_merit_list == 2 * [1.0]
