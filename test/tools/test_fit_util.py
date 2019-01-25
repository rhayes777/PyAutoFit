import numpy as np
import pytest

from autofit.tools import fit_util


class TestResiduals:

    def test__model_matches_data__residual_map_all_0s(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)

        assert (residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()

    def test__model_data_mismatch__no_masking__residual_map_non_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)

        assert (residual_map == np.array([-1.0, 0.0, 1.0, 2.0])).all()

    def test__model_data_mismatch__mask_included__masked_residual_map_set_to_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask,
                                                                           model_data=model_data)

        assert (residual_map == np.array([0.0, 0.0, 1.0, 0.0])).all()


class TestChiSquareds:

    def test__model_mathces_data__chi_sq_all_0s(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)

        assert (chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()

    def test__model_data_mismatch__no_masking__chi_sq_non_0(self):
        
        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)

        assert (chi_squared_map == np.array([(1.0 / 2.0)**2.0, 0.0, (1.0 / 2.0)**2.0, (2.0 / 2.0)**2.0])).all()

    def test__model_data_mismatch__mask_included__masked_chi_sqs_set_to_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)
        
        assert (chi_squared_map == np.array([0.0, 0.0, (1.0 / 2.0)**2.0, 0.0])).all()

    def test__model_data_mismatch__masked_noise_value_is_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        noise_map = np.array([2.0, 2.0, 2.0, 0.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)

        assert (chi_squared_map == np.array([0.0, 0.0, (1.0 / 2.0) ** 2.0, 0.0])).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_normalization(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask,
                                                                           model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map,
                                                                                        mask=mask,
                                                                                        noise_map=noise_map)
        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(mask=mask, chi_squared_map=chi_squared_map)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=mask, noise_map=noise_map)
        likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                  noise_normalization=noise_normalization)

        chi_squared = 0.0
        noise_normalization = np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0)) + \
                     np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0))

        assert likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__model_data_mismatch__no_masking__chi_squared_and_noise_normalization_are_lh(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)
        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(mask=mask, chi_squared_map=chi_squared_map)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=mask, noise_map=noise_map)
        likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                  noise_normalization=noise_normalization)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_squared = ((1.0 / 2.0)**2.0) + 0.0 + ((1.0 / 2.0)**2.0) + ((2.0 / 2.0)**2.0)
        noise_normalization = np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0)) + \
                     np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0))

        assert likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__same_as_above_but_different_noise_in_each_pixel(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([1.0, 2.0, 3.0, 4.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)
        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(mask=mask, chi_squared_map=chi_squared_map)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=mask, noise_map=noise_map)
        likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                  noise_normalization=noise_normalization)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_squared = 1.0 + (1.0 / (3.0**2.0)) + 0.25
        noise_normalization = np.log(2 * np.pi * (1.0**2.0)) + np.log(2 * np.pi * (2.0**2.0)) + \
                     np.log(2 * np.pi * (3.0**2.0)) + np.log(2 * np.pi * (4.0**2.0))

        assert likelihood == pytest.approx(-0.5 * (chi_squared + noise_normalization), 1e-4)

    def test__model_data_mismatch__mask_certain_pixels__lh_non_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        noise_map = np.array([1.0, 2.0, 3.0, 4.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)
        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(mask=mask, chi_squared_map=chi_squared_map)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=mask, noise_map=noise_map)
        likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                  noise_normalization=noise_normalization)

        # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

        chi_squared = 0.0 + (1.0 / 3.0)**2.0
        noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(2 * np.pi * (3.0**2.0))

        assert likelihood == pytest.approx(-0.5 * (chi_squared + noise_normalization), 1e-4)

    def test__model_data_mismatch__mask_certain_pixels__lh_non_0__2d_data(self):

        data = np.array([[10.0, 10.0],
                         [10.0, 10.0]])
        mask = np.array([[True, False],
                         [False, True]])
        noise_map = np.array([[1.0, 2.0],
                              [3.0, 4.0]])
        model_data = np.array([[11.0, 10.0],
                                [9.0, 8.0]])

        residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(residual_map=residual_map, mask=mask,
                                                                                        noise_map=noise_map)
        chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(mask=mask, chi_squared_map=chi_squared_map)
        noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(mask=mask, noise_map=noise_map)
        likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(chi_squared=chi_squared,
                                                                                  noise_normalization=noise_normalization)

        # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

        chi_squared = 0.0 + (1.0 / 3.0)**2.0
        noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(2 * np.pi * (3.0**2.0))

        assert likelihood == pytest.approx(-0.5 * (chi_squared + noise_normalization), 1e-4)