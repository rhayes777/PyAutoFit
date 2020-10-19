# %%
"""
Tutorial 5: Data and Models
===========================

Up to now, we've used used the `Aggregator` to load and inspect the `Samples` of 3 model-fits.

In this tutorial, we'll look at how the way we designed our source code makes it easy to use the `Aggregator` to
inspect, interpret and plot the results of the model-fit, including refitting the best models to our data.
"""

# %%
#%matplotlib inline

from pyprojroot import here
workspace_path = str(here())
%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from howtofit.chapter_2_results import src as htf


# %%
"""
To load these results with the `Aggregator`, we again point it to the path of the results we want it to inspect, with
our path straight to the `Aggregator` results ensuring we don't need to filter our `Aggregator` in this tutorial.
"""

# %%
agg = af.Aggregator(directory="output/howtofit/chapter_2/aggregator")

# %%
"""
We can use the `Aggregator` to load a generator of every fit`s dataset, by changing the `output` attribute to the 
`dataset` attribute at the end of the aggregator.

Note that in the source code for chapter 2, specifically in the `phase.py` module, we specified that the the `Dataset` 
object would be saved to hard-disk such that the `Aggregator` can load it.
"""

# %%
dataset_gen = agg.values("dataset")
print("Datasets:")
print(list(dataset_gen), "\n")

# %%
"""
It is here the object-oriented design of our `plot.py` module comes into its own. We have the `Dataset` objects loaded, 
meaning we can easily plot each `Dataset` using the `dataset_plot.py` module.
"""

# %%
for dataset in agg.values("dataset"):
    htf.plot.Dataset.data(dataset=dataset)

# %%
"""
The `Dataset` names are available as part of the `Dataset`.
"""

# %%
for dataset in agg.values("dataset"):
    print(dataset.name)

# %%
"""
The `info` dictionary we input into the `Phase` is also available.
"""

# %%
for info in agg.values("info"):
    print(info)

# %%
"""
We can repeat the same trick to get the `mask` of every fit.
"""

# %%
mask_gen = agg.values("mask")
print("Masks:")
print(list(mask_gen), "\n")


# %%
"""
We`re going to refit each `Dataset` with the `max_log_likelihood_instance` of each model-fit. To do this, we'll need 
each `Phase`'s `MaskedDataset`.

(If you are unsure what the `zip` is doing below, it essentially combines the `dataset_gen`, `mask_gen` and 
`settings_gen` into one list such that we can iterate over all three simultaneously to create each `MaskedDataset`).

The `MaskedDataset` may have been altered by the `data_trim_left` and `data_trim_right` `SettingsPhase`. We can 
load the `SettingsPhase` via the `Aggregator` to use these settings when we create the `MaskedDataset`.
"""

# %%
dataset_gen = agg.values("dataset")
mask_gen = agg.values("mask")
settings_gen = agg.values("settings")

masked_datasets = [
    htf.MaskedDataset(
        dataset=dataset, mask=mask, settings=settings.settings_masked_dataset
    )
    for dataset, mask, settings in zip(dataset_gen, mask_gen, settings_gen)
]

# %%
"""
There is a problem with how we set up the `MaskedDataset`s above, can you guess what it is?

We used lists! If we had fit a large sample of data, the above object would store the `MaskedDataset` of all objects
simultaneously in memory on our hard-disk, likely crashing our laptop! To avoid this, we must write functions that
manipulate the `Aggregator` generators as generators themselves. Below is an example function that performs the same
task as above.
"""

# %%
def masked_dataset_from_agg_obj(agg_obj):

    dataset = agg_obj.dataset
    mask = agg_obj.mask
    settings = agg_obj.settings

    return htf.MaskedDataset(
        dataset=dataset, mask=mask, settings=settings.settings_masked_dataset
    )


# %%
"""
To manipulate this function as a generator using the `Aggregator`, we apply it to the `Aggregator`'s `map` function.

The `masked_dataset_gen` below ensures that we avoid representing all `MaskedDataset`'s simultaneously in memory.
"""

# %%
masked_dataset_gen = agg.map(func=masked_dataset_from_agg_obj)
print(list(masked_dataset_gen))

# %%
"""
Lets get the `max_log_likelihood_instance`s, as we did in tutorial 3.
"""

# %%
instances = [samps.max_log_likelihood_instance for samps in agg.values("samples")]

# %%
"""
Okay, we want to inspect the fit of each `max_log_likelihood_instance`. To do this, we reperform each fit.

First, we need to create the `model_data` of every `max_log_likelihood_instance`. Lets begin by creating a list 
of profiles of every phase.
"""

# %%
profiles = [instance.profiles for instance in instances]

# %%
"""
We can use these to create the `model_data` of each set of profiles (which in this case is just 1 `Gaussian`, but had
we included more profiles in the model would consist of multiple `Gaussian`s / `Exponential`s.).
"""

# %%
model_datas = [
    profile.gaussian.profile_from_xvalues(xvalues=dataset.xvalues)
    for profile, dataset in zip(profiles, agg.values("dataset"))
]

# %%
"""
And, as we did in tutorial 2, we can combine the `MaskedDataset`s and `model_data`s in a `Fit` object to create the
maximum likelihood fit of each phase!
"""

# %%
fits = [
    htf.FitDataset(masked_dataset=masked_dataset, model_data=model_data)
    for masked_dataset, model_data in zip(masked_datasets, model_datas)
]

# %%
"""
We can now plot different components of the `Fit` (again benefiting from how we set up the `fit_plots.py` module)!
"""

# %%
for fit in fits:
    htf.plot.FitDataset.residual_map(fit=fit)
    htf.plot.FitDataset.normalized_residual_map(fit=fit)
    htf.plot.FitDataset.chi_squared_map(fit=fit)

# %%
"""
Again, the code above does not use generators and could prove memory intensive for large datasets. Below is how we 
would perform the above task with generator functions, using the `masked_dataset_gen` above for the `MaskedDataset`.
"""

# %%
def model_data_from_agg_obj(agg_obj):

    xvalues = agg_obj.dataset.xvalues
    instance = agg_obj.samples.max_log_likelihood_instance
    profiles = instance.profiles

    return sum([profile.profile_from_xvalues(xvalues=xvalues) for profile in profiles])


def fit_from_agg_obj(agg_obj):

    masked_dataset = masked_dataset_from_agg_obj(agg_obj=agg_obj)
    model_data = model_data_from_agg_obj(agg_obj=agg_obj)

    return htf.FitDataset(masked_dataset=masked_dataset, model_data=model_data)


fit_gen = agg.map(func=fit_from_agg_obj)

for fit in fit_gen:
    htf.plot.FitDataset.residual_map(fit=fit)
    htf.plot.FitDataset.normalized_residual_map(fit=fit)
    htf.plot.FitDataset.chi_squared_map(fit=fit)

# %%
"""
Setting up the above objects (the `masked_dataset`s, `model data`s, `fit`s) was a bit of work. It wasn`t too many 
lines of code, but for something our users will want to do many times it`d be nice to have a short cut to setting them 
up, right?

In the source code module `aggregator.py` we've set up exactly such a short-cut. This module simply contains the 
generator functions above such that the generator can be created by passing the `Aggregator`. This provides us with 
convenience methods for quickly creating the `MaskedDataset`, `model_data` and `Fit`'s using a single line of code:
"""

# %%

masked_dataset_gen = htf.agg.masked_dataset_generator_from_aggregator(aggregator=agg)
model_data_gen = htf.agg.model_data_generator_from_aggregator(aggregator=agg)
fit_gen = htf.agg.fit_generator_from_aggregator(aggregator=agg)

htf.plot.FitDataset.residual_map(fit=list(fit_gen)[0])

# %%
"""
The methods in `aggregator.py` actually allow us to go one step further: they all us to create the `MaskedDataset` and
`Fit` objects using an input `SettingsMaskedDataset`. This means we can fit a `Dataset` with a `Phase` and then see how
the model-fits change if we customize the `Dataset` in different ways.

Below, we create and plot a `Fit` where the `MaskedDataset` is trimmed from the left and right.
"""

# %%
settings_masked_dataset = htf.SettingsMaskedDataset(
    data_trim_left=20, data_trim_right=20
)

fit_gen = htf.agg.fit_generator_from_aggregator(
    aggregator=agg, settings_masked_dataset=settings_masked_dataset
)

htf.plot.FitDataset.residual_map(fit=list(fit_gen)[0])

# %%
"""
For your model-fitting project, you`ll need to update the `aggregator.py` module in the same way. This is why we have 
emphasised the object-oriented design of our model-fitting project throughout. This design makes it very easy to 
inspect results via the `Aggregator` later on!
"""
