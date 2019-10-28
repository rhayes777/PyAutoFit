import autoarray as aa
import autoastro as am
import autofit as af

import os


def simulate_imaging_from_gaussian_and_output_to_fits(
    gaussian,
    pixel_scales,
    shape_2d,
    data_type,
    sub_size,
    psf=None,
    exposure_time=300.0,
    background_sky_level=1.0,
):

    # Setup the image-plane al.ogrid of the Imaging array which will be used for generating the image of the
    # simulated strong lens. A high-res sub-grid is necessary to ensure we fully resolve the central regions of the
    # lens and source galaxy light.
    grid = aa.grid.uniform(
        shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=sub_size
    )

    image = gaussian.profile_image_from_grid(grid=grid)

    # Simulate the Imaging data_type, remembering that we use a special image which ensures edge-effects don't
    # degrade our modeling of the telescope optics (e.al. the PSF convolution).
    imaging = aa.imaging.simulate(
        image=image,
        exposure_time=exposure_time,
        psf=psf,
        background_sky_level=background_sky_level,
        add_noise=True,
    )

    # Now, lets output this simulated imaging-simulate to the test_autoarray/simulate folder.
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

    data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=["simulate", data_type]
    )

    imaging.output_to_fits(
        image_path=data_path + "image.fits",
        psf_path=data_path + "psf.fits",
        noise_map_path=data_path + "noise_map.fits",
        overwrite=True,
    )

    aa.plot.imaging.subplot(
        imaging=imaging,
        output_filename="imaging",
        output_path=data_path,
        output_format="png",
    )

    aa.plot.imaging.individual(
        imaging=imaging,
        should_plot_image=True,
        should_plot_noise_map=True,
        should_plot_psf=True,
        should_plot_signal_to_noise_map=True,
        output_path=data_path,
        output_format="png",
    )


def make__gaussian(sub_size):

    data_type = "gaussian"

    # This lens-only system has a Dev Vaucouleurs spheroid / bulge.

    gaussian = am.lp.SphericalGaussian(centre=(0.0, 0.0), intensity=0.1, sigma=2.0)

    simulate_imaging_from_gaussian_and_output_to_fits(
        gaussian=gaussian,
        pixel_scales=0.1,
        shape_2d=(50, 50),
        data_type=data_type,
        sub_size=sub_size,
    )
