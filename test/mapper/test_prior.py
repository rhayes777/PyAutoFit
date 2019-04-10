from autofit.mapper import prior as p


class TestConstant(object):
    def test_truthy(self):
        assert not p.Constant(None)
