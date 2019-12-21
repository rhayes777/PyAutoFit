# PyAutoFit

**PyAutoFit** is a Python-based probablistic programming language that enables contemporary Bayesian inference techniques to be straightforwardly integrated into scientific modeling software. 

In contrast to libraries such as [PyMC3](https://github.com/pymc-devs/pymc3) and [STAN](https://github.com/stan-dev/stan), **PyAutoFit** specializes in  problems for fitting **very large-datasets** with **many different models**, with advanced functionality including **transdimensional model-fitting**.

## API Overview

**PyAutoFit** interfaces with Python classes and non-linear sampling packages such as [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html). Lets take a two-dimensional Gaussian as our moodel:

```python
class Gaussian(object):

    def __init__(
        self,
        centre = (0.0, 0.0), # <- PyAutoFit recognises these constructor arguments are the model
        intensity = 0.1,     # <- parameters of Gaussian profile.
        sigma = 0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma
```
**PyAutoFit** recognises that this Gaussian may be treated as a model component whose parameters could be fitted for by a non-linear search. To fit this Gaussian to some data we can create and run a **PyAutoFit** phase: 

```python
import autofit as af

# To perform the analysis we set up a phase, which takes a Gaussian class as the 
model & fits its parameters using a non-linear search (below, MultiNest).
phase = al.PhaseImaging(
    phase_name="example/phase_example",
    model=af.CollectionPriorModel(gaussian_0=af.Gaussian),
    optimizer_class=af.MultiNest,
)

# We pass a dataset to the phase, fitting it with the model above.
phase.run(dataset=dataset)
```

By interfacing with model components as Python classes, **PyAutoFit** takes care of the 'heavy lifting' that comes with   performing the fit, for example parametrizing the model, interfacing with the non-linear search and on-the-fly output and visusalization of the model-fit.

## Features

# Aggregation

Lets pretend we performed the Gaussian fit above to 100 indepedent data-sets. All **PyAutoFit** outputs contain metadata that enables them to be immediately loaded via the **aggregator** in Python script or Jupyter notebook:


```python
output_folder = "/path/to/gaussian_x100_fits/"
phase_name = "phase_example"

# First, we create an instance of the aggregator, which takes the output path as input, telling it where to load
# results from.
aggregator = af.Aggregator(
    directory=str(output_folder)
)

# We can get the output of every non-linear search of every data-set fitted using the phase above.

non_linear_outputs = aggregator.filter(
    phase=phase_name
).output
```

For fits to large data-sets **PyAutoFit** thus provides the tools necessary to feasibly inspect and interpret the large quantity of results. If many different models are fitted to a data-set, the aggregator provides tools to filter the results loaded.

# Model Mapping

When a problem 

The benefits of using **PyAutoFit** are:

- **Model Mapping** - Interfacing with Python class

**PyAutoFit** specializes in advanced model-fitting problems, where highly complex models with many plausible model paramertizations are fitted. **PyAutoFit** breaks the model-fitting procedure into a series of **linked non-linear searches**, or 'phases', where the results of earlier phases initialize the fitting of more complex models in later phases.

This allows **transdimensional model-fitting pipelines** to be built that enable fitting of extremely complex and high dimensional models to be reduced to a series of bite-sized model fits, such that even the most complex model fitting task can be **fully automated**. 

## Transdimensional Modeling

and allows automated transdimensional model-fitting pipelines for large data-sets to be written, by acting as an interface between Python classes

## Python Example

We will illustrate this with an example fitting two 2D Gaussians:

![alt text](https://github.com/rhayes777/PyAutoFit/blob/master/gaussian_example.png)

We are going to fit each Gaussian with a 2D Gaussian pofile. Traditional methods would both Gaussians simultaneously, making parameter space more complex, slower to sample and increasing the risk that we fail to locate the global maxima solution. With **PyAutoFit** we can instead build a transdimensional model fitting pipeline which breaks the the analysis down into 3 phases:

1) Fit only the left Gaussian.
2) Fit only the right Gaussian, using the model of the left Gaussian from phase 1 to improve their deblending.
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear optimizer searches parameter space.


```python
import autofit as af

def make_pipeline():

    # In phase 1, we will fit the Gaussian on the left.

    phase1 = af.Phase(
        phase_name="phase_1__left_gaussian",
        gaussians=af.CollectionPriorModel(gaussian_0=af.profiles.SphericalGaussian),
        optimizer_class=af.MultiNest,
    )

    # In phase 2, we will fit the Gaussian on the right, where the best-fit Gaussian resulting from phase 1 
    # above fits the left-hand Gaussian.

    phase2 = af.Phase(
        phase_name="phase_2__right_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
            gaussian_0=phase1.result.instance.gaussians.gaussian_0, # <- Use the Gaussian fitted in phase 1
            gaussian_1=gaussian_1,
        ),
        optimizer_class=af.MultiNest,
    )

    # In phase 3, we fit both Gaussians, using the results of phases 1 and 2 to initialize their model parameters.

    phase3 = af.Phase(
        phase_name="phase_3__both_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
            gaussian_0=phase1.result.model.gaussians.gaussian_0, # <- use phase 1 Gaussian results.
            gaussian_1=phase2.result.model.gaussians.gaussian_1, # <- use phase 2 Gaussian results.
        ),
        optimizer_class=af.MultiNest,
    )

    return toy.Pipeline(pipeline_name, phase1, phase2, phase3)
```

Of course, fitting two Gaussians is a fairly trivial model-fitting problem that does not require **PyAutoFit**. Nevertheless, the example above illustrates how one can break a model-fitting task down with **PyAutoFit**, an approach which is crucial for the following software packages: 

- [PyAutoLens](https://github.com/Jammy2211/PyAutoLens) - Software for fitting galaxy-galaxy strong gravitational lensing systems. In this example, a 5-phase **PyAutoFit** pipeline performs strong lens modeling using 10 different model components producing models with 20-40 parameters.

## Features

Advanced statistical modeling features in **PyAutoFit** include:

- **Model Mapping** - Interface with Python classes to define and fit complex models parameterized with many different model components.
- **Pipelines** - Write transdimensional analysis pipelines to fit complex models to large data-sets in a fully automated way.
- **Non-linear Optimizers** - Combine a variety of non-linear search techniques (e.g. gradient descent, nested sampling, MCMC).
- **Aggregation** - Model results are stored in a database format that enables quick manipulate of large sets of results for inspection and interpretation.

## Future

The following features are planned for 2020:

- **Generalized Linear Models** - After fitting a large suite of data fit for global trends in the **PyAutoFit** model results.
- **Hierarchical modeling** - Combine fits over multiple data-sets to perform hierarchical inference.
- **Time series modelling** - Fit temporally varying models using fits which marginalize over time.
- **Approximate Bayesian Computational** - Likelihood-free modeling.
- **Transdimensional Sampling** - Sample non-linear parameter spaces with variable numbers of model components and parameters.

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
