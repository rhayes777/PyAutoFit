import itertools

import pytest

import autofit as af

from autofit.text import formatter as frm


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


def test_parameterization():
    model = af.Collection(collection=af.Collection(gaussian=af.Model(af.ex.Gaussian)))

    parameterization = model.parameterization
    assert parameterization == (
        """model                                                                           Collection (N=3)
    collection                                                                  Collection (N=3)
        gaussian                                                                Gaussian (N=3)"""
    )


def test_root():
    model = af.Model(af.ex.Gaussian)
    parameterization = model.parameterization
    assert parameterization == (
        "model                                                                           Gaussian (N=3)"
    )


def test_instance():
    model = af.Collection(collection=af.Collection(gaussian=af.ex.Gaussian()))

    parameterization = model.parameterization
    assert parameterization == (
        """model                                                                           Collection (N=0)
    collection                                                                  Collection (N=0)
        gaussian                                                                Gaussian (N=0)"""
    )


def test_tuple_prior():
    centre = af.TuplePrior()
    centre.centre_0 = af.UniformPrior()
    centre.centre_1 = af.UniformPrior()

    model = af.Model(af.ex.Gaussian, centre=centre)
    parameterization = model.parameterization
    assert parameterization == (
        "model                                                                           Gaussian (N=4)"
    )


@pytest.fixture(name="formatter")
def make_info_dict():
    formatter = frm.TextFormatter(line_length=20, indent=4)
    formatter.add(("one", "one"), 1)
    formatter.add(("one", "two"), 2)
    formatter.add(("one", "three", "four"), 4)
    formatter.add(("three", "four"), 4)

    return formatter


class TestGenerateModelInfo:
    def test_info_string(self, formatter):
        ls = formatter.list

        assert ls[0] == "one"
        assert len(ls[1]) == 21
        assert ls[1] == "    one             1"
        assert ls[2] == "    two             2"
        assert ls[3] == "    three"
        assert ls[4] == "        four        4"
        assert ls[5] == "three"
        assert ls[6] == "    four            4"

    def test_basic(self):
        mm = af.ModelMapper()
        mm.mock_class = af.m.MockClassx2
        model_info = mm.info

        assert (
            model_info
            == """Total Free Parameters = 2

model                                                                           ModelMapper (N=2)
    mock_class                                                                  MockClassx2 (N=2)

mock_class
    one                                                                         UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
    two                                                                         UniformPrior [1], lower_limit = 0.0, upper_limit = 2.0"""
        )

    def test_with_instance(self):
        mm = af.ModelMapper()
        mm.mock_class = af.m.MockClassx2

        mm.mock_class.two = 1.0

        model_info = mm.info

        assert (
            model_info
            == """Total Free Parameters = 1

model                                                                           ModelMapper (N=1)
    mock_class                                                                  MockClassx2 (N=1)

mock_class
    one                                                                         UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
    two                                                                         1.0"""
        )

    def test_with_tuple(self):
        mm = af.ModelMapper()
        mm.tuple = (0, 1)

        assert (
            mm.info
            == """Total Free Parameters = 0

model                                                                           ModelMapper (N=0)

tuple                                                                           (0, 1)"""
        )

    # noinspection PyUnresolvedReferences
    def test_tuple_instance_model_info(self, mapper):
        mapper.mock_cls = af.m.MockChildTuplex2
        info = mapper.info

        mapper.mock_cls.tup_0 = 1.0

        assert len(mapper.mock_cls.tup.instance_tuples) == 1
        assert len(mapper.mock_cls.instance_tuples) == 1

        assert len(info.split("\n")) == len(mapper.info.split("\n"))


def test_integer_attributes():
    model = af.Model(af.ex.Gaussian)

    model.centre = 2

    print(model.info)

    assert (
        model.info
        == """Total Free Parameters = 2

model                                                                           Gaussian (N=2)

centre                                                                          2
normalization                                                                   UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
sigma                                                                           UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0"""
    )
