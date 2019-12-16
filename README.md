# PyAutoFit

**PyAutoFit** is a Python-based probablistic programming language that enables contemporary Bayesian inference techniques to be straightforwardly integrated into scientific modeling software. **PyAutoFit** allows automated transdimensional model-fitting pipelines for large data-sets to be written, by acting as an interface between Python classes and non-linear sampling packages such as [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html), 

**PyAutoFit** specializes in advanced model-fitting problems, where highly complex models with many plausible model paramertizations are fitted. **PyAutoFit** breaks the model-fitting procedure into a series of **linked non-linear searches**, or 'phases', where the results of earlier phases initialize the fitting of more complex models in later phases.

This allows **transdimensional model-fitting pipelines** to be built that enable fitting of extremely complex and high dimensional models to be reduced to a series of bite-sized model fits, such that even the most complex model fitting task can be **fully automated**. 

## Python Example

We will illustrate this with an example fitting two 2D Gaussians:

We are going to fit each Gaussian with a 2D Gaussian pofile. Traditional methods would both Gaussians simultaneously, making parameter space more complex, slower to sample and increasing the risk that we fail to locate the global maxima solution. With **PyAutoFit** we can instead build a transdimensional model fitting pipeline which breaks the the analysis down into 3 phases:

1) Fit only the left Gaussian.
2) Fit only the right Gaussian, using the model of the left Gaussian from phase 1 to improve their deblending.
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear optimizer searches parameter space.

**PyAutoFit** determines the components of a model by interacting with Python classes. For this example we use the SphericalGaussian class:

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

Of course, fitting two Gaussians is a fairly trivial model-fitting problem that does not require **PyAutoFit**. Nevertheless, the example above illustrates how one can break a model-fitting task down with **PyAutoFit**, an approach which is crucial for the following software packages: 

- **PyAutoLens(https://github.com/Jammy2211/PyAutoLens)** - Software for fitting galaxy-galaxy strong gravitational lensing systems. In this example, a 5-phase **PyAutoFit** pipeline performs strong lens modeling using 10 different model components producing models with 20-40 parameters.

## Features

Advanced statistical modeling features in **PyAutoFit** include:

- **Model Mapping** - Interface with Python classes to define and fit complex models parameterized with many different model components.
- **Pipelines** - Write transdimensional analysis pipelines to fit complex models to large data-sets in a fully automated way.
- **Non-linear Optimizers** - Combine a variety of non-linear search techniques (e.g. gradient descent, nested sampling, MCMC).
- **Aggregation** - **PyAutoFit** output are stored in a database format that enables quick manipulate of large sets of results for result inspection and interpretation.

## Future

The following features are planned for 2020:

- **Generalized Linear Models** - After fitting a large suite of data fit for global trends in the **PyAutoFit** model results.
- **Hierarchical modeling** - Combine fits over a range of data-sets to perform hierarchical inference.
- **Time series modelling** - Fit temporally varying models using bespoke model-fits which marginalize over the fit as a function of time.
- **Transdimensional Sampling** - Sample non-linear parameter spaces with model numbers of model components and parameters.

## Yet Another Probablistic Programming Language?

There already exist many options for incorporating Bayesian inference techniques into model fitting problems, such as PyMC3[https://github.com/pymc-devs/pymc3] and STAN[https://github.com/stan-dev/stan]. These packages allow simple models to be quickly defined, parametrized and fitted to data.

## Slack

We're building a **PyAutoFit** community on Slack, so you should contact us on our [Slack channel](https://pyautofit.slack.com/) before getting started. Here, I will give you the latest updates on the software & discuss how best to use **PyAutoFit** for your science case.

Unfortunately, Slack is invitation-only, so first send me an [email](https://github.com/Jammy2211) requesting an invite.

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
