import autofit as af


class MyResult(af.mock.MockResult):
    pass


class MyAnalysis(af.Analysis):
    def __init__(self):
        self.is_modified_before = False
        self.is_modified_after = False

    def log_likelihood_function(self, instance):
        pass

    def make_result(
        self,
        samples_summary,
        paths,
        samples=None,
        search_internal=None,
        analysis=None,
    ):
        return MyResult(samples=samples)

    def modify_before_fit(self, paths, model):
        self.is_modified_before = True
        return self

    def modify_after_fit(self, paths, model, result):
        self.is_modified_after = True
        return self


def test_result_type():
    model = af.Model(af.Gaussian)

    analysis = MyAnalysis().with_model(model)

    result = analysis.make_result(None, None)

    assert isinstance(result, MyResult)
