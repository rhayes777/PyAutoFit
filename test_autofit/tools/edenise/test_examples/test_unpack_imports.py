import pytest

from autofit.tools.edenise import File, Import
from autofit.tools.edenise import LineItem


@pytest.fixture(
    name="aliased_import"
)
def make_aliased_import():
    return Import.parse_fragment(
        "import autofit as af"
    )


def test_alias_import(
        aliased_import
):
    assert aliased_import.is_aliased


def test_alias(
        aliased_import
):
    assert aliased_import.alias == "af"


def test_not_alias_import():
    assert Import.parse_fragment(
        "import autofit"
    ).is_aliased is False


@pytest.fixture(
    name="file"
)
def make_file(
        package,
        examples_directory
):
    return File(
        examples_directory / "unpack_imports.py",
        parent=package,
        prefix=""
    )


def test_convert_if_numpy(
        package,
        examples_directory
):
    file = File(
        examples_directory / "unpack_numpy.py",
        parent=package,
        prefix=""
    )
    assert file.target_string == """
import numpy as np
if np.isnan(1):
    assert False
"""


def test_alias_imports(
        file
):
    alias_import, = file.aliased_imports
    assert alias_import.alias == "af"
    assert file.aliases == ["af"]


def test_attributes(
        file
):
    assert len(list(file.attributes())) == 3


def test_attributes_for_alias(
        file
):
    assert file.attributes_for_alias(
        "af"
    ) == {
               "Model",
               "Gaussian"
           }


def test_convert_import(file):
    import_from = Import.parse_fragment(
        """import autofit as af""",
        parent=file
    )
    assert import_from.as_from_import(
        attribute_names={
            "Model", "Gaussian"
        }
    ).target_string == """
from VIS_CTI_Autofit import Gaussian, Model
"""


def test_convert_numpy_import(file):
    import_from = Import.parse_fragment(
        """import numpy as np""",
        parent=file
    )
    assert import_from.as_from_import(
        attribute_names={
            "isnan"
        }
    ).target_string == """
from numpy import isnan
"""


def test_whole_file(
        file
):
    assert file.target_string == """
from VIS_CTI_Autofit import Gaussian, Model
model = Model(Gaussian)
print(model.prior_count)
"""


def test_replace_alias(
        file
):
    item = LineItem.parse_fragment(
        """model = af.Model(
    af.Gaussian
)
        """,
        parent=file
    )

    assert item.target_string == """
model = Model(Gaussian)
"""
