# PyAutoFit

**PyAutoFit** is a Python-based probablistic programming language that allows Bayesian inference techniques to be straightforwardly integrated into scientific modeling software. **PyAutoFit** specializes in:

- **Black box** models with complex and expensive likelihood functions. 
- Fitting **many different model parametrizations** to a data-set. 
- Modeling **extremely large-datasets** with a homogenous fitting procedure. 
- Automating complex model-fitting tasks via **transdimensional model-fitting pipelines**.

# API Overview

**PyAutoFit** interfaces with Python classes and non-linear sampling packages such as [PyMultiNest](http://johannesbuchner.github.io/pymultinest-tutorial/install.html). Lets take a two-dimensional Gaussian as our moodel:

```python
class Gaussian:

    def __init__(
        self,
        centre = (0.0, 0.0), # <- PyAutoFit recognises these constructor arguments are the model
        intensity = 0.1,     # <- parameters of the Gaussian.
        sigma = 0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma
```
**PyAutoFit** recognises that this Gaussian may be treated as a model component whose parameters could be fitted for by a non-linear search. To fit this Gaussian to some data we can create and run a **PyAutoFit** phase: 

```python
import autofit as af

# To perform the analysis we set up a phase, which takes the Gaussian as the 
# model & fits its parameters using the non-linear search MultiNest.

phase = al.PhaseImaging(
    phase_name="example/phase_example",
    model=af.CollectionPriorModel(gaussian_0=af.Gaussian),
    optimizer_class=af.MultiNest,
)

# We pass a dataset to the phase, fitting it with the model above.

phase.run(dataset=dataset)
```

By interfacing with Python classes **PyAutoFit** takes care of the 'heavy lifting' that comes with parametrizing and fitting the model. This includes interfacing with the non-linear search, storing results in structured directories and providing on-the-fly output and visusalization.

# Features

## Model Customization

**PyAutoFit** makes it straight forward to parameterize, customize and fit models made of multiple components. Below, we extend the example above to include a second Gaussian, with user-specified priors and a centre aligned with the first Gaussian:

```python
import autofit as af

# The model can be setup with multiple classes and before passing it to a phase
# we can customize the model parameters.

model = af.CollectionPriorModel(gaussian_0=af.Gaussian, gaussian_1=af.Gaussian)

# This aligns the centres of the two Gaussian, reducing the number of free parameters by 2.
model.gaussian_0.centre = model.gaussian_1.centre

# This fixes the first Gaussian's sigma value to 0.5, reducing the number of free parameters by 1.
model.gaussian_0.sigma = 0.5

# We can customize the priors on any model parameter.
model.gaussian_0.intensity = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
model.gaussian_1.sigma = af.UniformPior(lower_limit=0.0, upper_limit=2.0)
model.gaussian_1.sigma = af.GaussianPrior(mean=0.1, sigma=0.05)

phase = al.PhaseImaging(
    phase_name="example/phase_example",
    model=model,
    optimizer_class=af.MultiNest,
)
```

## Aggregation

For fits to large data-sets **PyAutoFit** provides tools to manipulate the vast library of results output. 

Lets pretend we performed the Gaussian fit above to 100 indepedent data-sets. Every **PyAutoFit** output contains metadata meaning that we can immediately load it via the **aggregator** into a Python script or Jupyter notebook:


```python
output_path = "/path/to/gaussian_x100_fits/" # <- There is 100 separate fits in this folder.
phase_name = "phase_example"

# First, we create an instance of the aggregator, which uses the output path to load results.

aggregator = af.Aggregator(
    directory=str(output_path)
)

# We can get the output of every non-linear search of every data-set fitted using the phase above.

non_linear_outputs = aggregator.filter(
    phase=phase_name
).output

# From here, we can inspect results as we please, for example printing all 100 most likely models.

print([output.most_likely_instance from output in non_linear_outputs]

```

If many different phases are used to perform different model-fits to a data-set, the aggregator provides tools to filter out results.

## Transdimensional Modeling

In transdimensional modeling many different models are paramertized and fitted to the same data-set.  

This is performed using **transdimensional model-fitting pipelines**, which break the model-fit into a series of **linked non-linear searches**, or phases. Initial phases fit simplified realizations of the model, whose results are used to initialize fits using more complex models in later phases. 

Fits of complex models with large dimensionality can therefore be broken down into a series of **bite-sized model fits**, allowing even the most complex model fitting problem to be **fully automated**. 

Lets illustrate this with an example fitting two 2D Gaussians:

![alt text](https://github.com/rhayes777/PyAutoFit/blob/master/gaussian_example.png)

We're going to fit each with the 2D Gaussian profile above. Traditional approaches would fit both Gaussians simultaneously, making parameter space more complex, slower to sample and increasing the risk that we fail to locate the global maxima solution. With **PyAutoFit** we can instead build a transdimensional model fitting pipeline which breaks the the analysis down into 3 phases:

1) Fit only the left Gaussian.
2) Fit only the right Gaussian, using the model of the left Gaussian from phase 1 to reduce blending.
3) Fit both Gaussians simultaneously, using the results of phase 1 & 2 to initialize where the non-linear optimizer searches parameter space.


```python
import autofit as af

def make_pipeline():

    # In phase 1, we will fit the Gaussian on the left.

    phase1 = af.Phase(
        phase_name="phase_1__left_gaussian",
        gaussians=af.CollectionPriorModel(gaussian_0=af.profiles.Gaussian),
        optimizer_class=af.MultiNest,
    )

    # In phase 2, we will fit the Gaussian on the right, where the best-fit Gaussian 
    # resulting from phase 1 above fits the left-hand Gaussian.

    phase2 = af.Phase(
        phase_name="phase_2__right_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
            gaussian_0=phase1.result.instance.gaussians.gaussian_0, # <- Use the Gaussian fitted in phase 1
            gaussian_1=gaussian_1,
        ),
        optimizer_class=af.MultiNest,
    )

    # In phase 3, we fit both Gaussians, using the results of phases 1 and 2 to 
    # initialize their model parameters.

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

Althoguh somewhat trivial, this example illustrates how easily a model-fit can be broken down with **PyAutoFit**.

[PyAutoLens](https://github.com/Jammy2211/PyAutoLens) shows a real-use case of transdimensional modeling, fitting galaxy-scale strong gravitational lenses. In this example pipeline, a 5-phase **PyAutoFit** pipeline breaks-down the fit of 5 diferent models composed of over 10 unique model components and 10-30 free parameters.

## Future

The following features are planned for 2020:

- **Bayesian Model Comparison** - Determine the most probable model via the Bayesian evidence.
- **Generalized Linear Models** - Fit for global trends to model fits to large data-sets.
- **Hierarchical modeling** - Combine fits over multiple data-sets to perform hierarchical inference.
- **Time series modelling** - Fit temporally varying models using fits which marginalize over time.
- **Approximate Bayesian Computation** - Likelihood-free modeling.
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
