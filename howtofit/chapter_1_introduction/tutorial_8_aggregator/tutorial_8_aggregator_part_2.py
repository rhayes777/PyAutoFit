# %%
"""
Tutorial 8: Aggregator Part 2
=============================

In part 1 of tutorial 8, we fitted 3 datasets and used the aggregator to load their results. We focused on the
results of the non-linear search, Emcee. In part 2, we'll look at how the way we designed our source code
makes it easy to use these results to plot results and data.
"""

# %%
#%matplotlib inline

from autoconf import conf
import autofit as af
from howtofit.chapter_1_introduction.tutorial_8_aggregator import (
    src as htf,
)

from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/howtofit/config",
    output_path=f"{workspace_path}/howtofit/output",
)

# %%
"""
To load these results with the aggregator, we again point it to the path of the results we want it to inspect.
"""

# %%
output_path = f"{workspace_path}/howtofit/output"

agg = af.Aggregator(directory=str(output_path))
phase_name = "phase_t8"
agg_filter = agg.filter(agg.phase == phase_name)

# %%
"""
We can use the aggregator to load a generator of every fit's dataset, by changing the 'output' attribute to the 
'dataset' attribute at the end of the aggregator. We'll filter by phase name again to get datasets of only the fits 
performed for this tutorial.

Note that we had to manually specify in the 'phase.py' for the dataset to be saved too hard-disk such that the 
aggregator can load it.
"""

# %%
dataset_gen = agg_filter.values("dataset")
print("Datasets:")
print(list(dataset_gen), "\n")

# %%
"""
It is here the object-based design of our plot module comes into its own. We have the dataset objects loaded, meaning
we can easily plot each dataset using the 'dataset_plot.py' module.
"""

# %%
for dataset in agg_filter.values("dataset"):
    htf.plot.Dataset.data(dataset=dataset)

# %%
"""
The dataset names are available, either as part of the dataset or via the aggregator's dataset_names method.
"""

# %%
for dataset in agg_filter.values("dataset"):
    print(dataset.name)

# %%
"""
The info dictionary we input into the pipeline is also available.
"""

# %%
for info in agg_filter.values("info"):
    print(info)

# %%
"""
We can repeat the same trick to get the mask of every fit.
"""

# %%
mask_gen = agg_filter.values("mask")
print("Masks:")
print(list(mask_gen), "\n")


# %%
"""
We're going to refer to our datasets using the best-fit model of each phase. To do this, we'll need each phase's masked
dataset.

(If you are unsure what the 'zip' is doing below, it essentially combines the'datasets' and 'masks' lists in such
a way that we can iterate over the two simultaneously to create each MaskedDataset).

The masked dataset may have been altered by the data_trim_ custom phase settings. We can load the meta_dataset via the 
aggregator to use these settings when we create the masked dataset.
"""

# %%
dataset_gen = agg_filter.values("dataset")
mask_gen = agg_filter.values("mask")

masked_datasets = [
    htf.MaskedDataset(dataset=dataset, mask=mask)
    for dataset, mask in zip(dataset_gen, mask_gen)
]

masked_datasets = [
    masked_dataset.with_left_trimmed(data_trim_left=meta_dataset.data_trim_left)
    for masked_dataset, meta_dataset in zip(
        masked_datasets, agg_filter.values("meta_dataset")
    )
]
masked_datasets = [
    masked_dataset.with_right_trimmed(data_trim_right=meta_dataset.data_trim_right)
    for masked_dataset, meta_dataset in zip(
        masked_datasets, agg_filter.values("meta_dataset")
    )
]

# %%
"""
There is a problem with how we set up the masked datasets above, can you guess what it is?

We used lists! If we had fit a large sample of data, the above object would store the masked dataset of all objects
simultaneously in memory on our hard-disk, likely crashing our laptop! To avoid this, we must write functions that
manipulate the aggregator generators as generator themselves. Below is an example function that performs the same
task as above.
"""

# %%
def masked_dataset_from_agg_obj(agg_obj):

    dataset = agg_obj.dataset
    mask = agg_obj.mask

    masked_dataset = htf.MaskedDataset(dataset=dataset, mask=mask)

    meta_dataset = agg_obj.meta_dataset

    masked_dataset = masked_dataset.with_left_trimmed(
        data_trim_left=meta_dataset.data_trim_left
    )
    masked_dataset = masked_dataset.with_right_trimmed(
        data_trim_right=meta_dataset.data_trim_right
    )

    return masked_dataset


# %%
"""
To manipulate this function as a generator using the aggregator, we must apply it to the aggregator's map function.

The masked_dataset_generator below ensures that we avoid representing all masked datasets simultaneously in memory.
"""

# %%
masked_dataset_gen = agg_filter.map(func=masked_dataset_from_agg_obj)
print(list(masked_dataset_gen))

# %%
"""
Lets get the the maximum likelihood model instances, as we did in part 1.
"""

# %%
instances = [
    samps.max_log_likelihood_instance for samps in agg_filter.values("samples")
]

# %%
"""
Okay, we want to inspect the fit of each best-fit model. To do this, we reperform each fit.

First, we need to create the model-data of every best-fit model instance. Lets begin by creating a list of profiles of
every phase.
"""

# %%
profiles = [instance.profiles for instance in instances]

# %%
"""
We can use these to create the model data of each set of profiles (Which in this case is just 1 Gaussian, but had
we included more profiles in the model would consist of multiple Gaussians / Exponentials).
"""

# %%
model_datas = [
    profile.gaussian.profile_from_xvalues(xvalues=dataset.xvalues)
    for profile, dataset in zip(profiles, agg_filter.values("dataset"))
]

# %%
"""
And, as we did in tutorial 2, we can combine the masked_datasets and model_datas in a Fit object to create the
maximum likelihood fit of each phase!
"""

# %%
fits = [
    htf.FitDataset(masked_dataset=masked_dataset, model_data=model_data)
    for masked_dataset, model_data in zip(masked_datasets, model_datas)
]

# %%
"""
We can now plot different components of the fit (again benefiting from how we set up the 'fit_plots.py' module)!
"""

# %%
for fit in fits:
    htf.plot.FitDataset.residual_map(fit=fit)
    htf.plot.FitDataset.normalized_residual_map(fit=fit)
    htf.plot.FitDataset.chi_squared_map(fit=fit)

# %%
"""
Again, the code above does not use generators and could prove memory intensive for large datasets. Below is how we 
would perform the above task with generator functions, using the masked_dataset_generator above for the masked 
dataset.
"""

# %%
def model_data_from_agg_obj(agg_obj):
    xvalues = agg_obj.dataset.xvalues
    instance = agg_obj.samples.max_log_likelihood_instance
    profiles = instance.profiles
    model_data = sum(
        [profile.profile_from_xvalues(xvalues=xvalues) for profile in profiles]
    )

    return model_data


def fit_from_agg_obj(agg_obj):
    masked_dataset = masked_dataset_from_agg_obj(agg_obj=agg_obj)
    model_data = model_data_from_agg_obj(agg_obj=agg_obj)

    return htf.FitDataset(masked_dataset=masked_dataset, model_data=model_data)


fit_gen = agg_filter.map(func=fit_from_agg_obj)

for fit in fit_gen:
    htf.plot.FitDataset.residual_map(fit=fit)
    htf.plot.FitDataset.normalized_residual_map(fit=fit)
    htf.plot.FitDataset.chi_squared_map(fit=fit)

# %%
"""
Setting up the above objects (the masked_datasets, model datas, fits) was a bit of work. It wasn't too many
lines of code, but for something we'll likely want to do many times it'd be nice to have a short cut to setting
them up, right?

In 'aggregator.py' we've set up exactly such a short-cut. This module simply contains the generator functions above 
such that the generator can be created by passing the aggregator. This provides us with convenience methods for quickly 
creating the masked dataset, model data and fits using single lines of code:
"""

# %%

from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.phase import aggregator

masked_dataset_gen = aggregator.masked_dataset_generator_from_aggregator(
    aggregator=agg_filter
)
model_data_gen = aggregator.model_data_generator_from_aggregator(aggregator=agg_filter)
fit_gen = aggregator.fit_generator_from_aggregator(aggregator=agg_filter)

# %%
"""
For your model-fitting project, you'll need to update the 'aggregator.py' module in the same way. This is why we have 
emphasised the object-oriented design of our model-fitting project through. This design makes it very easy to inspect 
results via the aggregator later on!
"""
