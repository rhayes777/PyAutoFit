# %%
"""
Tutorial 10: Data and Models
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

from os import path
import autofit as af


# %%
"""
To load these results with the `Aggregator`, we again point it to the path of the results we want it to inspect, with
our path straight to the `Aggregator` results ensuring we don't need to filter our `Aggregator` in this tutorial.
"""

# %%
agg = af.Aggregator(
    directory=path.join("output", "howtofit", "chapter_1", "aggregator")
)

# %%
"""
We'll reuse the `plot_line` function of previous tutorials, however it now displays to the notebook as opposed to
outputting the results to a .png file.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt


def plot_line(xvalues, line, title=None, ylabel=None, errors=None, color="k"):
    plt.errorbar(
        x=xvalues, y=line, yerr=errors, color=color, ecolor="k", elinewidth=1, capsize=2
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()


# %%
"""
We can use the `Aggregator` to load a generator of every fit`s data, by changing the `output` attribute to the 
`data` attribute at the end of the aggregator.

Note that in the `Analysis` class of tutorial 7, we specified that the `data` object would be saved to hard-disc using
the `save_attributes_for_aggregator` method, so that the `Aggregator` can load it.
"""

# %%
data_gen = agg.values("data")
print("Datas:")
print(list(data_gen), "\n")

# %%
"""We can plot the `data` using the `plot_line` method."""

# %%
for data in agg.values("data"):
    plot_line(
        xvalues=np.arange(data.shape[0]),
        line=data,
        title="Data",
        ylabel="Data Values",
        color="k",
    )

# %%
"""The `info` dictionary we input into the `NonLinearSearch` is also available."""

# %%
for info in agg.values("info"):
    print(info)

# %%
"""
We can repeat the same trick to get the `noise_map` of every fit.
"""

# %%
noise_map_gen = agg.values("noise_map")
print("Masks:")
print(list(noise_map_gen), "\n")


# %%
"""
We're going to refit each dataset with the `max_log_likelihood_instance` of each model-fit and plot the residuals.

(If you are unsure what the `zip` is doing below, it essentially combines the `data_gen`, `noise_map_gen` and
 `samples_gen` into one list such that we can iterate over them simultaneously).
"""

# %%
samples_gen = agg.values("samples")
data_gen = agg.values("data")
noise_map_gen = agg.values("noise_map")

for data, noise_map, samples in zip(data_gen, noise_map_gen, samples_gen):

    instance = samples.max_log_likelihood_instance

    xvalues = np.arange(data.shape[0])

    model_data = sum(
        [profile.profile_from_xvalues(xvalues=xvalues) for profile in instance]
    )

    residual_map = data - model_data

    plot_line(
        xvalues=xvalues,
        line=residual_map,
        title="Residual Map",
        ylabel="Residuals",
        color="k",
    )

# %%
"""
There is a problem with how we plotted the residuals above, can you guess what it is?

We used lists! If we had fit a large sample of data, the above object would store the data of all objects simultaneously 
in memory on our hard-disk, likely crashing our laptop! To avoid this, we must write functions that manipulate the 
`Aggregator` generators as generators themselves. Below is an example function that performs the same task as above.
"""

# %%
def plot_residuals_from_agg_obj(agg_obj):

    data = agg_obj.data
    noise_map = agg_obj.noise_map
    samples = agg_obj.samples

    instance = samples.max_log_likelihood_instance

    xvalues = np.arange(data.shape[0])

    model_data = sum(
        [profile.profile_from_xvalues(xvalues=xvalues) for profile in instance]
    )

    residual_map = data - model_data

    plot_line(
        xvalues=xvalues,
        line=residual_map,
        title="Residual Map",
        ylabel="Residuals",
        color="k",
    )


# %%
"""
To manipulate this function as a generator using the `Aggregator`, we apply it to the `Aggregator`'s `map` function.
"""

# %%
plot_residuals_gen = agg.map(func=plot_residuals_from_agg_obj)

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
