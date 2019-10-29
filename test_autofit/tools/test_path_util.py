import os
import shutil
import autofit as af

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))

test_data_path = "{}/../test_files/arrays/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestMakeAndReturnPath:
    def test__1_directory_input__makes_directory__returns_path(self):
        path = af.path_util.make_and_return_path_from_path_and_folder_names(
            path=test_data_path, folder_names=["test1"]
        )

        assert path == test_data_path + "test1/"
        assert os.path.exists(path=test_data_path + "test1")

        shutil.rmtree(test_data_path + "test1")

    def test__multiple_directories_input__makes_directory_structure__returns_full_path(
        self
    ):
        path = af.path_util.make_and_return_path_from_path_and_folder_names(
            path=test_data_path, folder_names=["test1", "test2", "test3"]
        )

        assert path == test_data_path + "test1/test2/test3/"
        assert os.path.exists(path=test_data_path + "test1")
        assert os.path.exists(path=test_data_path + "test1/test2")
        assert os.path.exists(path=test_data_path + "test1/test2/test3")

        shutil.rmtree(test_data_path + "test1")
