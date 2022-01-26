import pytest

from autofit.tools.edenise import File


@pytest.fixture(
    name="file"
)
def make_file(
        package,
        examples_directory
):
    return File(
        examples_directory / "local_import.py",
        parent=package,
        prefix=""
    )


def test(file):
    print(file.target_string)
    assert file.target_string == '''

def func():
    from VIS_CTI_Autofit.VIS_CTI_NonLinear.VIS_CTI_Grid.VIS_CTI_GridSearch import GridSearch
'''
