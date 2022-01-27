import pytest

from autofit.tools.edenise import File


@pytest.fixture(
    name="function_file"
)
def make_function_file(
        package,
        examples_directory
):
    return File(
        examples_directory / "local_import.py",
        parent=package,
        prefix=""
    )


def test_function(function_file):
    print(function_file.target_string)
    assert function_file.target_string == '''

def func():
    from VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Grid.VIS_CTI_GridSearch import GridSearch
    from VIS_CTI_Autoconf.VIS_CTI_Tools.decorators import CachedProperty
'''


@pytest.fixture(
    name="class_file"
)
def make_class_file(
        package,
        examples_directory
):
    return File(
        examples_directory / "class_import.py",
        parent=package,
        prefix=""
    )


def test_class(class_file):
    print(class_file.target_string)
    assert class_file.target_string == '''

class MyClass():

    @property
    def slim(self):
        """
        Return a `Grid1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size, 2].

        If it is already stored in its `slim` representation  the `Grid1D` is returned as it is. If not, it is
        mapped from  `native` to `slim` and returned as a new `Grid1D`.
        """
        from VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Grid.VIS_CTI_GridSearch import GridSearch
        from VIS_CTI_Autoconf.VIS_CTI_Tools.decorators import CachedProperty
'''
