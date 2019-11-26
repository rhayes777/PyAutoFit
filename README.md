# PyAutoFit

**PyAutoFit** is a Python software framework that enables contemporary Bayesian inference techniques to be applied to large data-sets in a fully automated way. Acting as an interface between Python classes and non-linear sampling packages such as [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html), **PyAutoFit** allows transdimensional model fitting pipelines to be built for general modeling problems.

[The key thing about **PyAutoFit** is that it uses multiplle NLOs.]

## Python Example

In this example, we perform galaxy photometry on two merging galaxies by fitting each with an elliptial Sersic light profile:

Traditional methods would fit the light of both galaxies simultaneously, making parameter space more complex and slower to sample. With **PyAutoFit** we can break the analysis down into three phases:

1) Fit the light of the left galaxy.
2) Fit the light of the right galaxy, using the model of the left galaxy from phase 1 to better deblend their light.
3) Fit the light of both galaxies, using the results of phase 1 & 2 to initialize where the non-linear optimizer searches parameter space.

**PyAutoFit** determines the components of a model by interacting with Python classes. For this example we use the EllipticalSersic class:

```
class EllipticalSersic(object):

    def __init__(
            self,
            centre: tuple = (0.0, 0.0),     # <- PyAutoFit recognises these
            axis_ratio: float = 1.0,        #    constructor inputs as the
            phi: float = 0.0,               #    model parameters of the
            intensity: float = 0.1,         #    EllipticalSersic model.
            effective_radius: float = 0.6,
            sersic_index: float = 4.0,
    ):

        self.centre = centre
        self.axis_ratio = axis_ratio
        self.phi = phi
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index
```

This model, and its model parameters, are then used by PyAutoFit to build our 3 phase model-fitting pipeline:

```
import autofit as af

def make_pipeline():

    pipeline_name = "pipeline_example__fitting_multiple_galaxies"

    # In phase 1, we fit the main galaxy with a bulge + disk model, assuming that
    # not fitting the satellite won't significantly degrade the overall fit.

    phase1 = af.Phase(phase_name="phase_1__main_galaxy_fit",
            main_galaxy=af.PriorModel(
                bulge=af.toy.light_profiles.EllipticalSersic,
                disk=af.toy.light_profiles.EllipticalSersic),
        optimizer_class=af.MultiNest)

    # In phase 2, we fit the satellite galaxy's light. The best-fit model of the
    # main galaxy in phase 1 is used to subtract its light and cleanly reveal the
    # satellite for the fit. This information is passed using 'instance' term below.

    phase2 = af.Phase(phase_name="phase_2__satellite",
            main_galaxy=af.PriorModel(
                bulge=phase1.result.instance.main_galaxy.bulge,
                disk=phase1.result.instance.main_galaxy.disk),
            satellite_galaxy=af.PriorModel(
                light=af.toy.EllipticalSersic),
        optimizer_class=af.MultiNest)

    # In phase 3, we fit the light of both galaxies simultaneously using priors
    # derived from the results of phases 1 & 2 to begin sampling in the maximum
    # likelihood regions of parameter space. This information is passed using
    # the 'model' term below.

    phase3 = af.Phase(phase_name="phase_3__all_galaxies",
            main_galaxy=af.PriorModel(
                bulge=phase1.result.model.main_galaxy.bulge,
                disk=phase1.result.model.main_galaxy.disk),
            left_satellite=af.PriorModel(
                light=phase2.result.model.left_satellite.light),
        optimizer_class=af.MultiNest)

    return af.Pipeline(pipeline_name, phase1, phase2, phase3)
```

## Status

**PyAutoFit** is ready for adoption by model-fitting packages, and is currently used by [**PyAutoLens**](https://github.com/Jammy2211/PyAutoLens) for modeling strong gravitational lens galaxies.PyAutoFit

However, **PyAutoFit** is still currently in alpha and lacking many keey materials to help new user adoption (documentation, templates, etc.). Projects interested in using **PyAutoFit** should therefore contact us directly first about the best way to get started.

## Slack

We're building a **PyAutoFit** community on Slack, so you should contact us on our [Slack channel](https://pyautofit.slack.com/) before getting started. Here, I will give you the latest updates on the software & discuss how best to use **PyAutoFit** for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

## Features

**PyAutoFit's** advanced modeling features include:

- **Non-linear Optimizers** - Fit models to data using a variety of non-linear search techniques and algorithms.
- **Model Mapping** - Interface with Python classes to define a model and map non-linear optimizer samples to a model parameterization.
- **Pipelines** - Write transdimensional analysis pipelines to fit complex models to large data-sets in a fully automated way.

The following features are planned for the next year:

- **Hierarchical modeling** - Fit for global trends of a model within individual fits to a data-set, permitting more general inferences to be made.
- **Time series modelling** - Fit temporally varying models using bespoke model-fits which marginalize over the fit as a function of time.
- **Transdimensional Sampling** - Sample non-linear parameter spaces with model numbers of model components and parameters.

## Depedencies

**PyAutoFit** requires [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html).

## Installation with conda

We recommend installation using a conda environment as this circumvents a number of compatibility issues when installing **PyMultiNest**.

First, install [conda](https://conda.io/miniconda.html).

Create a conda environment:

```
conda create -n autofit python=3.7 anaconda
```

Activate the conda environment:

```
conda activate autofit
```

Install multinest:

```
conda install -c conda-forge multinest
```

Install autofit:

```
pip install autofit
```

## Installation with pip

Installation is also available via pip, however there are reported issues with installing **PyMultiNest** that can make installation difficult, see the file [INSTALL.notes](https://github.com/Jammy2211/PyAutoFit/blob/master/INSTALL.notes)

```
$ pip install autofit
```

## Support & Discussion

If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our [Slack channel](https://pyautofit.slack.com/).

## Contributing

If you have any suggestions or would like to contribute please get in touch.

## Credits

### Developers

[Richard Hayes](https://github.com/rhayes777) - Lead developer

[James Nightingale](https://github.com/Jammy2211) - Lead developer