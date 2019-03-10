
import os
import shutil

import numpy as np
import pytest

from autofit.tools import path_util

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))

test_data_path = "{}/../test_files/array/".format(os.path.dirname(os.path.realpath(__file__)))

class TestPhaseFoldersFromPipelineName:

    def test__phase_folders_is_none__returns_pipeline_name(self):

        phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=None,
                                                                                     pipeline_name='test')

        assert phase_folders == ['test']

    def test__phase_folders_are_not_none__returns_phase_folders_and_pipeline_name(self):

        phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=['folder1', 'folder2'],
                                                                                     pipeline_name='test')

        assert phase_folders == ['folder1', 'folder2', 'test']

class PathFromFolderNames:

    def test__1_directory_input__makes_directory__returns_path(self):

        path = path_util.path_from_folder_names(folder_names=['test1'])

        assert path == 'test1/'

    def test__multiple_directories_input__makes_directory_structure__returns_full_path(self):

        path = path_util.path_from_folder_names(folder_names=['test1', 'test2', 'test3'])

        assert path == 'test1/test2/test3/'

class TestMakeAndReturnPath:

    def test__1_directory_input__makes_directory__returns_path(self):

        path = path_util.make_and_return_path_from_path_and_folder_names(path=test_data_path, folder_names=['test1'])

        assert path == test_data_path + 'test1/'
        assert os.path.exists(path=test_data_path+'test1')

        shutil.rmtree(test_data_path+'test1')

    def test__multiple_directories_input__makes_directory_structure__returns_full_path(self):

        path = path_util.make_and_return_path_from_path_and_folder_names(path=test_data_path, folder_names=['test1', 'test2', 'test3'])

        assert path == test_data_path + 'test1/test2/test3/'
        assert os.path.exists(path=test_data_path+'test1')
        assert os.path.exists(path=test_data_path+'test1/test2')
        assert os.path.exists(path=test_data_path+'test1/test2/test3')

        shutil.rmtree(test_data_path+'test1')