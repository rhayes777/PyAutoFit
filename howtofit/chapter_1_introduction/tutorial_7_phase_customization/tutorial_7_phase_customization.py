# %%
"""
Tutorial 7: Phase Customization
===============================

In this tutorial, we`re going to add input parameters to a `Phase` object that customizes the analysis. we'll use the
specific example of two input parameters that trim our `Dataset` from the left and right before fitting it. This
example is somewhat trivial (we could achieve almost the same effect with masking), but it will serve to illustrate
`Phase` customization.

When we customize a `Phase`, we'll use `SettingsPhase` to perform `tagging`. Here, we `tag` the output
path of the `Phase`'s `Result``., such that every time a `Phase` is run with a different customization a new set of
unique `Result`'s are stored for those `PhaseSetting``.. For a given `Dataset` we are thus able to fit it multiple
times using different settings to compare the results.

These new features have led to an additional module in the `phase` package called `settings.py`, as well as extensions
to the `dataset.py` module. Before looking at these modules, lets first perform a series of `Emcee` fits to see how
they change the behaviour of **PyAutoFit**.
"""

# %%
#%matplotlib inline

from pyprojroot import here
workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import numpy as np

from howtofit.chapter_1_introduction.tutorial_7_phase_customization import (
    src as htf,
)

# %%
"""
We`re now going to perform multiple fits, where each fit trims the `Dataset` that is fitted.

To do this, we'll set up `Phase`_s using a new class called `SettingsMaskedDataset`, which contains the settings that 
customize how a `MaskedDataset` is created. This has two inputs, `data_trim_left` and `data_trim_right`:

- `data_trim_left`:

  The `Dataset`'s `image` and `noise_map` are trimmed and removed from the left (e.g. 1d index values from 0).
  
  For example, if the `Dataset` has shape (100,) and we set `data_trim_left=10`, the `MaskedDataset` that is fitted 
  will have shape (90,). The `mask` is trimmed in the same way.

- `data_trim_right`:

  This behaves the same as `data_trim_left`, however `data` is removed from the right (e.g. 1D index values from the
  shape[0] value of the 1D data).

For our first phase, we will omit both of these settings (by passing them as `None`) and perform the fit from tutorial
4 where we fit a single `Gaussian` profile to `data` composed of a single `Gaussian` (unlike tutorial 4, we'll use a
_CollectionPriorModel_ to do this).
"""

# %%
settings_masked_dataset = htf.SettingsMaskedDataset(
    data_trim_left=None, data_trim_right=None
)

settings = htf.SettingsPhase(settings_masked_dataset=settings_masked_dataset)

phase = htf.Phase(
    search=af.Emcee(path_prefix="howtofit/chapter_1", name="phase_t7"),
    profiles=af.CollectionPriorModel(gaussian=htf.profiles.Gaussian),
    settings=settings,
)

# %%
"""
Set up the`Dataset` and `mask`.
"""

# %%
dataset_path = "dataset/howtofit/chapter_1/gaussian_x1"
data = af.util.numpy_array_from_json(file_path=f"{dataset_path}/data.json")
noise_map = af.util.numpy_array_from_json(file_path=f"{dataset_path}/noise_map.json")

dataset = htf.Dataset(data=data, noise_map=noise_map)
mask = np.full(fill_value=False, shape=dataset.data.shape)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t6"
    " folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Okay, what happened differently in this _Phase? To begin, lets note the output directory:

`autofit_workspace/howtofit/chapter_1_introduction/tutorial_7_phase_customization/output/phase_t7/settings`

There is a small change to this directory compared to tutorial 6, there is a new folder `settings` within which the
results are stored. It`ll be clear why this is in a moment.

Next, we`re going to customize and run a phase using the `data_trim_left` and `data_trim_right` parameters. First, we 
create a `SettingsMaskedDataset` and `SettingsPhase` object using our input values of these parameters. 
"""

# %%
settings_masked_dataset = htf.SettingsMaskedDataset(
    data_trim_left=20, data_trim_right=30
)

settings = htf.SettingsPhase(settings_masked_dataset=settings_masked_dataset)

# %%
"""
We now create a new `Phase` with these settings and run it (note that we haven`t changed the `name` from 
`phase_t7`, which you might think would cause conflicts in the path the results are output to).
"""

# %%
phase = htf.Phase(
    search=af.Emcee(path_prefix="howtofit/chapter_1", name="phase_t7"),
    profiles=af.CollectionPriorModel(gaussian=htf.profiles.Gaussian),
    settings=settings,
)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t6"
    " folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
You`ll note the results are now in a slightly different directory to the fit performed above:

 `autofit_workspace/howtofit/chapter_1_introduction/tutorial_7_phase_customization/output/phase_example/settings__trim_left_20__trim_right_30`

By customizing the `PhaseSetting``., **PyAutoFit** has changed the output path using a `tag`. There are two reasons 
**PyAutoFit** does this:

 1) Tags describes the analysis, making it explicit what was done to the `Dataset` for the fit.

 2) Tags create a unique output path, allowing you to compare results of `Phase`'s that use different `SettingsPhase`. 
    Equally if you run multiple phases with different `PhaseSetting`'s this ensures the `NonLinearSearch` won't
    use results generated via a different analysis method.

You should now check out the `settings.py` and `dataset.py` modules, to see how we implemented this.


When reading through this tutorial`s example source code, you may have felt it was a bit clunky having multiple 
_Settings_ classes each of which was set up separately to customize the `MaskedDataset` or the `Phase`. 

For the source code, it actually is quite clunky and could certainly be refactored to make the source code more clean. 
However, through experience we have found this design creates a much better API for a user when choosing settings, 
which will be seen in the next tutorial. Thus, we recommend you adopt the same settings API for your project!


In this tutorial, the `PhaseSetting`'s changed the `MaskedDataset` that was fitted. However, `PhaseSetting`'s do not 
necessarily need to customize the `Dataset`. For example, they could control some aspect of the model, for example the 
precision with which an aspect of the model is numerically calculated. For more complex fitting procedures, they may 
control whether certain features are used, which when turned on / off reduce the accuracy of the model at the expense 
of greater computational run-time.

_PhaseSetting_s are project specific and it could well be your modeling problem is simple enough not to need them.
However, if it does, remember that `PhaseSetting`. are a powerful means to fit models in different ways and compare 
whether this changes the model inferred. In later chapters, we'll discuss more complex model-fitting procedures that 
use `fast` less accurate `PhaseSetting`'s to initialize the model-fit, but switch to slower more accurate 
_PhaseSetting_`s later on.
"""
