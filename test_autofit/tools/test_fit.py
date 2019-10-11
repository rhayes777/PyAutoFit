import numpy as np

from autofit.tools import fit


class TestDataFit:

    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        mask = np.array([False, False, False, False])

        model_data = np.array([1.0, 2.0, 3.0, 4.0])

        data_fit = fit.DataFit(
            data=data, noise_map=noise_map, mask=mask, model_data=model_data)

        assert (data_fit._data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit._noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit._signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit._mask == np.array([False, False, False, False])).all()
        assert (data_fit._model_data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit._residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (data_fit._normalized_residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (data_fit._chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert data_fit.chi_squared == 0.0
        assert data_fit.reduced_chi_squared == 0.0
        assert data_fit.noise_normalization == np.sum(
            np.log(2 * np.pi * noise_map ** 2.0))
        assert data_fit.likelihood == -0.5 * (
                data_fit.chi_squared + data_fit.noise_normalization)

    def test__image_and_model_mismatch__no_masking__check_values_are_correct(self):

        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        mask = np.array([False, False, False, False])

        model_data = np.array([1.0, 1.0, 1.0, 1.0])

        data_fit = fit.DataFit(
            data=data, noise_map=noise_map, mask=mask, model_data=model_data)

        assert (data_fit._data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit._noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit._signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit._mask == np.array([False, False, False, False])).all()
        assert (data_fit._model_data == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert (data_fit._residual_map == np.array([0.0, 1.0, 2.0, 3.0])).all()
        assert (data_fit._normalized_residual_map == np.array([0.0, (1.0 / 2.0), (2.0 / 2.0), (3.0 / 2.0)])).all()
        assert (data_fit._chi_squared_map == np.array(
            [0.0, (1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit.chi_squared == (1.0 / 2.0) ** 2.0 + (2.0 / 2.0) ** 2.0 + (
                3.0 / 2.0) ** 2.0
        assert data_fit.reduced_chi_squared == data_fit.chi_squared / 4.0
        assert data_fit.noise_normalization == np.sum(
            np.log(2 * np.pi * noise_map ** 2.0))
        assert data_fit.likelihood == -0.5 * (
                data_fit.chi_squared + data_fit.noise_normalization)

    def test__image_and_model_mismatch__include_masking__check_values_are_correct(self):

        data = np.array([1.0, 2.0, 3.0, 4.0])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        mask = np.array([False, True, True, False])

        model_data = np.array([1.0, 1.0, 1.0, 1.0])

        data_fit = fit.DataFit(data=data, noise_map=noise_map, mask=mask,
                                  model_data=model_data)

        assert (data_fit._data == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (data_fit._noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (data_fit._signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (data_fit._mask == np.array([False, True, True, False])).all()
        assert (data_fit._model_data == model_data).all()
        assert (data_fit._residual_map == np.array([0.0, 0.0, 0.0, 3.0])).all()
        assert (data_fit._normalized_residual_map == np.array([0.0, 0.0, 0.0, (3.0 / 2.0)])).all()
        assert (data_fit._chi_squared_map == np.array([0.0, 0.0, 0.0, (3.0 / 2.0) ** 2.0])).all()
        assert data_fit.chi_squared == (3.0 / 2.0) ** 2.0
        assert data_fit.reduced_chi_squared == data_fit.chi_squared / 2.0
        assert data_fit.noise_normalization == 2.0 * np.sum(
            np.log(2 * np.pi * 2.0 ** 2.0))
        assert data_fit.likelihood == -0.5 * (
                data_fit.chi_squared + data_fit.noise_normalization)

