import numpy as np
import pytest

from autofit.tools import fit


class TestDataFitter:

    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        mask = np.array([False, False, False, False])

        model_data = np.array([1.0, 2.0, 3.0, 4.0])

        data_fit = fit.DataFit(data=data, noise_map=noise_map, mask=mask, model_data=model_data)

        assert (data_fit.data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit.mask == np.array([False, False, False, False])).all()
        assert (data_fit.model_data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (data_fit.chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert data_fit.chi_squared == 0.0
        assert data_fit.reduced_chi_squared == 0.0
        assert data_fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert data_fit.likelihood == -0.5*(data_fit.chi_squared + data_fit.noise_normalization)

    def test__image_and_model_mismatch__no_masking__check_values_are_correct(self):

        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        mask = np.array([False, False, False, False])

        model_data = np.array([1.0, 1.0, 1.0, 1.0])

        data_fit = fit.DataFit(data=data, noise_map=noise_map, mask=mask, model_data=model_data)

        assert (data_fit.data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit.mask == np.array([False, False, False, False])).all()
        assert (data_fit.model_data == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (data_fit.residual_map == np.array([0.0, 1.0, 2.0, 3.0])).all()
        assert (data_fit.chi_squared_map == np.array([0.0, (1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit.chi_squared == (1.0 / 2.0) ** 2.0 + (2.0 / 2.0) ** 2.0 + (3.0 / 2.0) ** 2.0
        assert data_fit.reduced_chi_squared == data_fit.chi_squared / 4.0
        assert data_fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert data_fit.likelihood == -0.5*(data_fit.chi_squared + data_fit.noise_normalization)

    def test__image_and_model_mismatch__include_masking__check_values_are_correct(self):

        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        mask = np.array([False, True, True, False])

        model_data = np.array([1.0, 1.0, 1.0, 1.0])

        data_fit = fit.DataFit(data=data, noise_map=noise_map, mask=mask, model_data=model_data)

        assert (data_fit.data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit.mask == np.array([False, True, True, False])).all()
        assert (data_fit.model_data == model_data).all()
        assert (data_fit.residual_map == np.array([0.0, 0.0, 0.0, 3.0])).all()
        assert (data_fit.chi_squared_map == np.array([0.0, 0.0, 0.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit.chi_squared == (3.0 / 2.0) ** 2.0
        assert data_fit.reduced_chi_squared == data_fit.chi_squared / 2.0
        assert data_fit.noise_normalization == 2.0 * np.sum(np.log(2 * np.pi * 2.0 ** 2.0))
        assert data_fit.likelihood == -0.5*(data_fit.chi_squared + data_fit.noise_normalization)


class TestDataFitterStack:

    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        data_0 = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map_0 = np.array([2.0, 2.0, 2.0, 2.0])
        mask_0 = np.array([False, False, False, False])
        model_data_0 = np.array([1.0, 2.0, 3.0, 4.0])

        data_1 = np.array([4.0, 3.0, 2.0, 1.0])
        noise_map_1 = np.array([3.0, 3.0, 3.0, 3.0])
        mask_1 = np.array([False, False, False, False])
        model_data_1 = np.array([4.0, 3.0, 2.0, 1.0])

        data_fit = fit.DataFitStack(datas=[data_0, data_1], noise_maps=[noise_map_0, noise_map_1],
                               masks=[mask_0, mask_1], model_datas=[model_data_0, model_data_1])

        assert (data_fit.datas[0] == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.noise_maps[0] == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.masks[0] == np.array([False, False, False, False])).all()
        assert (data_fit.model_datas[0] == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.residual_maps[0] == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (data_fit.chi_squared_maps[0] == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert data_fit.chi_squareds[0] == 0.0
        assert data_fit.reduced_chi_squareds[0] == 0.0
        assert data_fit.noise_normalizations[0] == np.sum(np.log(2 * np.pi * noise_map_0 ** 2.0))
        assert data_fit.likelihoods[0] == -0.5 * (data_fit.chi_squareds[0] + data_fit.noise_normalizations[0])

        assert (data_fit.datas[1] == np.array([4.0, 3.0, 2.0, 1.0])).all()
        assert (data_fit.noise_maps[1] == np.array([3.0, 3.0, 3.0, 3.0])).all()
        assert (data_fit.masks[1] == np.array([False, False, False, False])).all()
        assert (data_fit.model_datas[1] == np.array([4.0, 3.0, 2.0, 1.0])).all()
        assert (data_fit.residual_maps[1] == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (data_fit.chi_squared_maps[1] == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert data_fit.chi_squareds[1] == 0.0
        assert data_fit.reduced_chi_squareds[1] == 0.0
        assert data_fit.noise_normalizations[1] == np.sum(np.log(2 * np.pi * noise_map_1 ** 2.0))
        assert data_fit.likelihoods[1] == -0.5 * (data_fit.chi_squareds[1] + data_fit.noise_normalizations[1])

        assert data_fit.chi_squared == 0.0
        assert data_fit.reduced_chi_squared == 0.0
        assert data_fit.noise_normalization == data_fit.noise_normalizations[0] + data_fit.noise_normalizations[1]
        assert data_fit.likelihood == -0.5 * (data_fit.chi_squared + data_fit.noise_normalization)

    def test__image_and_model_mismatch__no_masking__check_values_are_correct(self):

        data_0 = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map_0 = np.array([2.0, 2.0, 2.0, 2.0])
        mask_0 = np.array([False, False, False, False])
        model_data_0 = np.array([1.0, 1.0, 1.0, 1.0])

        data_1 = np.array([4.0, 3.0, 2.0, 1.0])
        noise_map_1 = np.array([3.0, 3.0, 3.0, 3.0])
        mask_1 = np.array([False, False, False, False])
        model_data_1 = np.array([2.0, 2.0, 2.0, 2.0])

        data_fit = fit.DataFitStack(datas=[data_0, data_1], noise_maps=[noise_map_0, noise_map_1],
                               masks=[mask_0, mask_1], model_datas=[model_data_0, model_data_1])

        assert (data_fit.datas[0] == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.noise_maps[0] == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.masks[0] == np.array([False, False, False, False])).all()
        assert (data_fit.model_datas[0] == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (data_fit.residual_maps[0] == np.array([0.0, 1.0, 2.0, 3.0])).all()
        assert (data_fit.chi_squared_maps[0] == np.array([0.0, (1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit.chi_squareds[0] == (1.0 / 2.0) ** 2.0 + (2.0 / 2.0) ** 2.0 + (3.0 / 2.0) ** 2.0
        assert data_fit.reduced_chi_squareds[0] == data_fit.chi_squareds[0] / 4.0
        assert data_fit.noise_normalizations[0] == np.sum(np.log(2 * np.pi * noise_map_0 ** 2.0))
        assert data_fit.likelihoods[0] == -0.5 * (data_fit.chi_squareds[0] + data_fit.noise_normalizations[0])

        assert (data_fit.datas[1] == np.array([4.0, 3.0, 2.0, 1.0])).all()
        assert (data_fit.noise_maps[1] == np.array([3.0, 3.0, 3.0, 3.0])).all()
        assert (data_fit.masks[1] == np.array([False, False, False, False])).all()
        assert (data_fit.model_datas[1] == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.residual_maps[1] == np.array([2.0, 1.0, 0.0, -1.0])).all()
        assert (data_fit.chi_squared_maps[1] == np.array([(2.0 / 3.0) ** 2.0, (1.0 / 3.0) ** 2.0, 0.0, (-1.0 / 3.0) ** 2.0])).all()
        assert data_fit.chi_squareds[1] == (2.0 / 3.0) ** 2.0 + (1.0 / 3.0) ** 2.0 + (-1.0 / 3.0) ** 2.0
        assert data_fit.reduced_chi_squareds[1] == data_fit.chi_squareds[1] / 4.0
        assert data_fit.noise_normalizations[1] == np.sum(np.log(2 * np.pi * noise_map_1 ** 2.0))
        assert data_fit.likelihoods[1] == -0.5 * (data_fit.chi_squareds[1] + data_fit.noise_normalizations[1])

        assert data_fit.chi_squared == data_fit.chi_squareds[0] + data_fit.chi_squareds[1]
        assert data_fit.reduced_chi_squared == data_fit.reduced_chi_squareds[0] + data_fit.reduced_chi_squareds[1]
        assert data_fit.noise_normalization == data_fit.noise_normalizations[0] + data_fit.noise_normalizations[1]
        assert data_fit.likelihood == -0.5 * (data_fit.chi_squared + data_fit.noise_normalization)

    def test__image_and_model_mismatch__include_masking__check_values_are_correct(self):

        data_0 = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map_0 = np.array([2.0, 2.0, 2.0, 2.0])
        mask_0 = np.array([False, True, True, False])
        model_data_0 = np.array([1.0, 1.0, 1.0, 1.0])

        data_1 = np.array([4.0, 3.0, 2.0, 1.0])
        noise_map_1 = np.array([3.0, 3.0, 3.0, 3.0])
        mask_1 = np.array([True, True, False, False])
        model_data_1 = np.array([2.0, 2.0, 2.0, 2.0])

        data_fit = fit.DataFitStack(datas=[data_0, data_1], noise_maps=[noise_map_0, noise_map_1],
                               masks=[mask_0, mask_1], model_datas=[model_data_0, model_data_1])

        assert (data_fit.datas[0] == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit.noise_maps[0] == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.masks[0] == np.array([False, True, True, False])).all()
        assert (data_fit.model_datas[0] == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (data_fit.residual_maps[0] == np.array([0.0, 0.0, 0.0, 3.0])).all()
        assert (data_fit.chi_squared_maps[0] == np.array([0.0, 0.0, 0.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit.chi_squareds[0] == (3.0 / 2.0) ** 2.0
        assert data_fit.reduced_chi_squareds[0] == data_fit.chi_squareds[0] / 2.0
        assert data_fit.noise_normalizations[0] == 2.0 * np.sum(np.log(2 * np.pi * 2.0 ** 2.0))
        assert data_fit.likelihoods[0] == -0.5 * (data_fit.chi_squareds[0] + data_fit.noise_normalizations[0])

        assert (data_fit.datas[1] == np.array([4.0, 3.0, 2.0, 1.0])).all()
        assert (data_fit.noise_maps[1] == np.array([3.0, 3.0, 3.0, 3.0])).all()
        assert (data_fit.masks[1] == np.array([True, True, False, False])).all()
        assert (data_fit.model_datas[1] == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit.residual_maps[1] == np.array([0.0, 0.0, 0.0, -1.0])).all()
        assert (data_fit.chi_squared_maps[1] == np.array([0.0, 0.0, 0.0, (-1.0 / 3.0) ** 2.0])).all()
        assert data_fit.chi_squareds[1] == (-1.0 / 3.0) ** 2.0
        assert data_fit.reduced_chi_squareds[1] == data_fit.chi_squareds[1] / 2.0
        assert data_fit.noise_normalizations[1] == 2.0 * np.sum(np.log(2 * np.pi * 3.0 ** 2.0))
        assert data_fit.likelihoods[1] == -0.5 * (data_fit.chi_squareds[1] + data_fit.noise_normalizations[1])

        assert data_fit.chi_squared == data_fit.chi_squareds[0] + data_fit.chi_squareds[1]
        assert data_fit.reduced_chi_squared == data_fit.reduced_chi_squareds[0] + data_fit.reduced_chi_squareds[1]
        assert data_fit.noise_normalization == data_fit.noise_normalizations[0] + data_fit.noise_normalizations[1]
        assert data_fit.likelihood == pytest.approx(-0.5 * (data_fit.chi_squared + data_fit.noise_normalization), 1.0e4)