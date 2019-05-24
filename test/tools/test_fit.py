import numpy as np
import pytest

from autofit.tools import fit


class TestDataFit:

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

class TestDataFit1d:

    def test__image_and_model_mismatch__include_masking__check_values_are_correct(self):

        data_1d = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map_1d = np.array([2.0, 2.0, 2.0, 2.0])
        mask_1d = np.array([False, True, True, False])
        model_data_1d = np.array([1.0, 1.0, 1.0, 1.0])

        data_fit_1d = fit.DataFit1D(data_1d=data_1d, noise_map_1d=noise_map_1d, mask_1d=mask_1d,
                                    model_data_1d=model_data_1d)

        assert (data_fit_1d.data_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit_1d.noise_map_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit_1d.signal_to_noise_map_1d == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit_1d.mask_1d == np.array([False, True, True, False])).all()
        assert (data_fit_1d.model_data_1d == model_data_1d).all()
        assert (data_fit_1d.residual_map_1d == np.array([0.0, 0.0, 0.0, 3.0])).all()
        assert (data_fit_1d.chi_squared_map_1d == np.array([0.0, 0.0, 0.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit_1d.chi_squared == (3.0 / 2.0) ** 2.0
        assert data_fit_1d.reduced_chi_squared == data_fit_1d.chi_squared / 2.0
        assert data_fit_1d.noise_normalization == 2.0 * np.sum(np.log(2 * np.pi * 2.0 ** 2.0))
        assert data_fit_1d.likelihood == -0.5*(data_fit_1d.chi_squared + data_fit_1d.noise_normalization)
