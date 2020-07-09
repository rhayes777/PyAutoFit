# %%
"""
Tutorial 7: Phase Customization
===============================

In this tutorial, we're going to add input parameters to a Phase object that customizes the analysis. We'll use the
specific example of two input parameters that trim our data-set from the left and right before fitting it. This
example is somewhat trivial (we could achieve almost the same effect with masking), but it will serve to illustrate
phase customization.

When we customize a phase, we'll use these 'phase settings' to perform phase tagging. Here, we 'tag' the output
path of the phase's results, such that every time a phase is run with a different customization a new set of
unique results are stored for those settings. For a given data-set we are thus able to fit it multiple times using
different settings to compare the results.

These new features have led to additional modules in the 'phase' package called 'meta_dataset.py' and 'settings.py'.
Before looking at these modules, lets first perform a series of Emcee fits to see how they change the behaviour
of PyAutoFit.
"""

# %%
#%matplotlib inline

from autoconf import conf
import autofit as af
import numpy as np

from howtofit.chapter_1_introduction.tutorial_7_phase_customization import (
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
We're now going to perform multiple fits, where each fit trims the data-set that is fitted.
To do this, we'll set up phases with the phase-settings 'data_trim_left' and 'data_trim_right'.

- data_trim_left:

  The dataset's image and noise-map are trimmed and removed from the left (e.g. 1d index values from 0).
  For example, if the dataset has shape (100,) and we set data_trim_left=10, the dataset that is fitted will have
  shape (90,). The mask is trimmed in the same way.

- data_trim_right:

  This behaves the same as data_trim_left, however data is removed from the right (e.g. 1D index values from the
  shape of the 1D data).

For our first phase, we will omit both the phase setting (by setting it to None) and perform the fit from tutorial
4 where we fit a single Gaussian profile to data composed of a single Gaussian (unlike tutorial 4, we'll use a
CollectionPriorModel to do this).
"""

# %%
phase = htf.Phase(
    phase_name="phase_t7",
    profiles=af.CollectionPriorModel(gaussian=htf.profiles.Gaussian),
    settings=htf.PhaseSettings(data_trim_left=None, data_trim_right=None),
    search=af.Emcee(),
)

# %%
"""
Import the simulator module, set up the Dataset and mask and set up the dataset.
"""

# %%
from howtofit.simulators.chapter_1 import gaussian_x1

dataset = htf.Dataset(data=gaussian_x1.data, noise_map=gaussian_x1.noise_map)
mask = np.full(fill_value=False, shape=dataset.data.shape)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t6"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Okay, lets look at what happened differently in this phase. To begin, lets note the output directory:

'autofit_workspace/howtofit/chapter_1_introduction/tutorial_7_phase_customization/output/phase_t7/settings'

There is a small change to this directory compared to tutorial 6, there is a new folder 'settings' within which the
results are stored. It'll be clear why this is in a moment.

Next, we're going to customize and run a phase using the data_trim_left and right parameters. First, we create a 
PhaseSettings object using our input values of these parameters. 
"""

# %%
settings = htf.PhaseSettings(data_trim_left=20, data_trim_right=30)

# %%
"""
We now create a new phase with these settings and run it (note that we haven't changed the phase_name from 'phase_t7', 
which you might think would cause conflicts in the path the results are output to).
"""

# %%
phase = htf.Phase(
    phase_name="phase_t7",
    profiles=af.CollectionPriorModel(gaussian=htf.profiles.Gaussian),
    settings=settings,
    search=af.Emcee(),
)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t6"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
You'll note the results are now in a slightly different directory to the fit performed above:

 'autofit_workspace/howtofit/chapter_1_introduction/tutorial_7_phase_customization/output/phase_example/settings__trim_left_20__trim_right_30'

By customizing the phase's settings, PyAutoFit has changed it output path using a tag for this phase. There are two
reasons PyAutoFit does this:

 1) Tags describes the analysis, making it explicit what was done to the dataset for the fit.

 2) Tags create a unique output path, allowing you to compare results of phases that use different settings. Equally,
 if you run multiple phases with different settings this ensures the non-linear search (e.g. Emcee) won't
 inadvertantly use results generated via a different analysis method.

You should now check out the 'settings.py' and 'meta_dataset.py' modules in the 'phase' package, to see how we 
implemented this.

In this tutorial, the phase setting changed the data-set that was fitted. However, phase settings do not necessarily
need to customize the data-set. For example, they could control some aspect of the model, for example the precision
by which the model image is numerically calculated. For more complex fitting procedures, settings could control
whether certain features are used, which when turned on / off reduce the accuracy of the model at the expensive of
greater computational run-time.

Phase settings are project specific and it could well be your modeling problem is simple enough not to need them.
However, if it does, remember that phase settings are a powerful means to fit models using different settings and
compare whether a setting does or does not change the model inferred. In later chapters, we'll discuss more complex
model-fitting procedures that could use 'fast' less accurate settings to initialize the model-fit, but switch to
slower more accurate settings later on.
"""
