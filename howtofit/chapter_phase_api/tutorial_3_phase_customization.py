# %%
"""
Tutorial 3: Phase Customization
===============================

In this tutorial, we're going to use the `Settings` object of our `Phase` object to customize the analysis. we'll use
the specific example of two input parameters that trim our `Dataset` from the left and right before fitting it. This
example is somewhat trivial, but it will serve to illustrate `Phase` settings and customization.

When we customize a `Phase`, we'll use `SettingsPhase` to perform `tagging`. Here, we `tag` the output
path of the `Phase`'s `Result`'s, such that every time a `Phase` is run with a different customization a new set of
unique `Result`'s are stored for those `PhaseSetting`'s. For a given `Dataset` we are thus able to fit it multiple
times using different settings to compare the results.

These new features have led to an additional module in the `phase` package called `settings.py`, as well as extensions
to the `dataset.py` module. Before looking at these modules, lets first perform a series of `Emcee` fits to see how
they change the behaviour of **PyAutoFit**.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

# %%
"""We again need to tell PyAutoFit where the project config files are."""

# %%
from os import path
from autoconf import conf

conf.instance.push(
    new_path=path.join(workspace_path, "howtofit", "chapter_phase_api", "src", "config")
)

import autofit as af

import src as htf

# %%
"""
We're now going to perform multiple fits, where each fit trims the `Dataset` that is fitted.

To do this, we'll set up `Phase`'s using a new class called `SettingsDataset`, which contains the settings that 
customize how a `Dataset` is created. This has two inputs, `data_trim_left` and `data_trim_right`:

- `data_trim_left`:

  The `Dataset`'s `image` and `noise_map` are trimmed and removed from the left (e.g. 1d index values from 0).
  
  For example, if the `Dataset` has shape (100,) and we set `data_trim_left=10`, the `Dataset` that is fitted 
  will have shape (90,). The `mask` is trimmed in the same way.

- `data_trim_right`:

  This behaves the same as `data_trim_left`, however `data` is removed from the right (e.g. 1D index values from the
  shape[0] value of the 1D data).

For our first phase, we will omit both of these settings (by passing them as `None`) and perform the fit using a 
single `Gaussian` profile.
"""

# %%
settings_dataset = htf.SettingsDataset(data_trim_left=None, data_trim_right=None)

settings = htf.SettingsPhase(settings_dataset=settings_dataset)

phase = htf.Phase(
    search=af.Emcee(
        path_prefix=path.join("howtofit", "chapter_phase_api"), name="phase_t3"
    ),
    settings=settings,
    profiles=af.CollectionPriorModel(gaussian=htf.Gaussian),
)

# %%
"""Set up the`Dataset`."""

# %%
dataset_path = path.join("dataset", "howtofit", "chapter_phase_api", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

dataset = htf.Dataset(data=data, noise_map=noise_map)

print(
    "Emcee has begun running, checkout:\n "
    "autofit_workspace/output/howtofit/chapter_phase_api/phase_t3\n"
    "for live output of the results.\n"
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!\n"
)

phase.run(dataset=dataset)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Okay, what happened differently in this `Phase`? To begin, lets note the output directory:

`autofit_workspace/output/howtofit/chapter_phase_api/phase_t3/settings`

There is a small change to this directory compared to chapter 1, there is a new folder `settings` within which the
results are stored. It`ll be clear why this is in a moment.

Next, we're going to customize and run a phase using the `data_trim_left` and `data_trim_right` parameters. First, we 
create a `SettingsDataset` and `SettingsPhase` object using our input values of these parameters. 
"""

# %%
settings_dataset = htf.SettingsDataset(data_trim_left=20, data_trim_right=30)

settings = htf.SettingsPhase(settings_dataset=settings_dataset)

# %%
"""
We now create a new `Phase` with these settings and run it (note that we haven`t changed the `name` from 
`phase_t3`, which you might think would cause conflicts in the path the results are output to).
"""

# %%
phase = htf.Phase(
    search=af.Emcee(
        path_prefix=path.join("howtofit", "chapter_phase_api"), name="phase_t3"
    ),
    settings=settings,
    profiles=af.CollectionPriorModel(gaussian=htf.Gaussian),
)

print(
    "Emcee has begun running, checkout:\n "
    "autofit_workspace/output/howtofit/chapter_phase_api/phase_t3\n"
    "for live output of the results.\n"
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!\n"
)

phase.run(dataset=dataset)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
You'll note the results are now in a slightly different directory to the fit performed above:

 `autofit_workspace/output/howtofit/chapter_phase_api/phase_t3/settings__dataset[trim_left_20__trim_right_30]`

By customizing the `PhaseSetting`'s, **PyAutoFit** has changed the output path using a `tag`. There are two reasons 
**PyAutoFit** does this:

 1) Tags describes the analysis, making it explicit what was done to the `Dataset` for the fit.

 2) Tags create a unique output path, allowing you to compare results of `Phase`'s that use different `SettingsPhase`. 
    Equally, if you run multiple phases with different `PhaseSetting`'s this ensures the `NonLinearSearch` won't
    use results generated via a different analysis method.

You should now check out the `settings.py` and `dataset.py` modules, to see how we implemented this.


When reading through this tutorial's example source code, you may have felt it was a bit clunky having multiple 
`Settings` classes each of which was set up separately to customize the `Dataset` or the `Phase`. 

In terms of the source code, it actually is quite clunky and could certainly be refactored to make the source code 
more clean. However, through experience we have found this design creates a much better API for a user when choosing 
settings, which will be emphasized seen in the next tutorial. 

For your model-fitting project, these `Settings` objects may customize more than just the `Dataset`, but perhaps 
details of how the model is fitted, or a mask applied to the data, or something else entirely. We recommend you adopt 
the settings API of the template project for your project!
"""
