# %%
"""
Tutorial 5: Visualization & Masking
===================================

In the previous tutorial, we wrote source code which used **PyAutoFit** to fit a 1D `Gaussian` model to a `Dataset`.
In this tutorial, we'll extend this source code`s phase package to perform a number of additional tasks:

 - Masking: The phase is passed a mask such that regions of the `Dataset` are omitted by the `log_likelihood_function`.

 - Visualization: Images showing the model fit are output on-the-fly during the `NonLinearSearch`.

These new features have lead to an additional module in the `phase` package not present in tutorial 4, called
`visualizer.py`. Before looking at this module, lets perform a fit to see the changed behaviour of **PyAutoFit**.
"""

# %%
#%matplotlib inline

from pyprojroot import here
workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from howtofit.chapter_1_introduction.tutorial_5_visualization_masking import (
    src as htf,
)

import numpy as np

# %%
"""
Load the data and set up the `Dataset`.
"""

# %%
dataset_path = "dataset/howtofit/chapter_1/gaussian_x1"
data = af.util.numpy_array_from_json(file_path=f"{dataset_path}/data.json")
noise_map = af.util.numpy_array_from_json(file_path=f"{dataset_path}/noise_map.json")

dataset = htf.Dataset(data=data, noise_map=noise_map)

# %%
"""
Before fitting data, we may want to mask it, removing regions of the data we know are defective or where there is no
signal.

To facilitate this we have added a new class to the module `dataset.py`. This takes our `Dataset` and a mask and 
combines the two to create a `MaskedDataset`. The `fit.py` module has also been updated to use a mask during the fit. 
Check them both out now to see how the mask is used! 

As mentioned in tutorial 4, if your model-fitting problem involves fitting masked data, you should be able to copy and 
paste the `fit.py` module for your own project.

Masking occurs within the phase package of **PyAutoFit**, which we'll inspect at the end of the tutorial. However,
for a phase to run it now requires that a `mask` is passed to it. For this tutorial, lets create a which removes the
last 30 data-points in our `data`.

(In our convention, a `mask` value of `True` means it IS masked and thus removed from the fit).
"""

# %%
mask = np.full(fill_value=False, shape=dataset.data.shape)
mask[-30:] = True

# %%
"""
Lets now perform the fit from tutorial 4, but with a `MaskedDataset` and visualization.
"""

# %%
phase = htf.Phase(
    search=af.Emcee(path_prefix="howtofit/chapter_1", name="phase_t5"),
    gaussian=af.PriorModel(htf.Gaussian),
)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t4"
    " folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

# %%
"""
Note that we are passing our `mask` to the `phase.run` function, which we did not in previous tutorials.
"""

# %%
phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Lets check that this phase performs visualization. Navigate to the folder `image` in the directory above. You should 
see a set of folders containing the visualization of the `Dataset` and the `Fit`. As promised, our phase is
now taking care of the visualization of our model.

Visualization happens `on-the-fly`, such that during `Emcee` these images are output using the current maximum 
likelihood model `Emcee` has found. For models more complex than our 1D `Gaussian` this is useful, as it means we can 
check that `Emcee` has found reasonable solutions during a run and can thus cancel it early if it has ended up with an
incorrect solution.

How often does **PyAutoFit** output new images? This is set by `visualize_every_update` in the config file
`config/visualize/general.ini`.

Finally, now inspect the `phase.py`, `analysis.py` and `visualizer.py` scripts in the source code. These describe how 
the `MaskedData` is set up and how visualization is performed.

And with that, we have completed this (fairly short) tutorial. There are two things worth ending on:

 1) In tutorial 4, we introduced the `plot` package that had functions specific to plotting attributes of
 a data-set and fit. This project structure has again helped us, by making it straight forward to perform plotting with 
 the _Visualizer-. For your model-fitting project you should aim to adhere to performing all plots in a `plot` 
 module, as more benefits will become clear in chapter 2!
    
 2) For our very simple 1D case, we used a 1D ndarray to represent a `mask`. For projects with more complicated
 datasets, one may require more complicated masks, warranting a `mask` package, `mask.py` module and `Mask2D` class. In 
 tutorial 9 we will show an example of this.
"""
