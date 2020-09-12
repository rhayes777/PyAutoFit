# %%
"""
Tutorial 2: Dataset Sample
==========================

In this tutorial, we'll fit a multiple datasets with the same phase, producing multiple sets of results on our
hard-disk. In the following tutorials, we then use these results and the _Aggregator_ to load the results into
our Jupyter notebook to interpret, inspect and plot the output results.

We'll fit 3 different dataset's, each with a single _Gaussian_ model.
"""

# %%
#%matplotlib inline

from autoconf import conf
import autofit as af
from howtofit.chapter_2_results import src as htf

import numpy as np
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our non-linear search.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/howtofit/config",
    output_path=f"{workspace_path}/howtofit/output/chapter_2",
)

# %%
"""
Here, for each _Dataset_ we are going to set up the correct path, load it, create its mask and fit it using a phase.

We want our results to be in a folder specific to the _Dataset_. We'll use the _Dataset_'s name string to do this. Lets
create a list of all 3 of our _Dataset_ names.

We'll also pass these names to the _Dataset_ when we create it - the name will be used by the _Aggregator_ to name the 
file the data is stored. More importantly, the name will be accessible to the aggregator, and we will use it to label 
figures we make via the aggregator.
"""

# %%
from howtofit.simulators.chapter_2 import (
    gaussian_x1_0,
    gaussian_x1_1,
    gaussian_x1_2,
)

dataset_names = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]
datas = [gaussian_x1_0.data, gaussian_x1_1.data, gaussian_x1_2.data]
noise_maps = [gaussian_x1_0.noise_map, gaussian_x1_1.noise_map, gaussian_x1_2.noise_map]

# %%
"""
We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the data of osbervation) that isn't part of the model-fit is made accessible to 
the _Aggregator_. For example, below we write info on the _Dataset_'s data of observation and exposure time.
"""

# %%
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

# %%
"""
This for loop runs over every _Dataset_, checkout the comments below for how we set up the path structure.
"""

# %%
for index in range(len(datas)):

    """The code below creates the _Dataset_ and mask as per usual."""

    dataset = htf.Dataset(data=datas[index], noise_map=noise_maps[index])

    mask = np.full(fill_value=False, shape=dataset.data.shape)

    """
    Here, we create a phase as normal. However, we also include an input parameter 'folders'. The phase folders
    define the names of folders that the phase goes in. For example, if a phase goes to the path:

        '/path/to/autofit_workspace/output/phase_name/'

    A phase folder with the input 'phase_folder' edits this path to:

        '/path/to/autofit_workspace/output/phase_folder/phase_name/'

    You can input multiple phase folders, for example 'folders=['folder_0', 'folder_1'] would create the path:

        '/path/to/autofit_workspace/output/folder_0/folder_1/phase_name/'

    Below, we use the data_name, so our results go in a folder specific to the _Dataset_, e.g:

        '/path/to/autofit_workspace/output/gaussian_x1_0/phase_t2/'
    """

    print(
        f"Emcee has begun running - checkout the "
        f"autofit_workspace/howtofit/chapter_2_results/output/aggregator/{dataset_names[index]}/phase_t2 folder for live "
        f"output of the results. This Jupyter notebook cell with progress once Emcee has completed - this could take a "
        f"few minutes!"
    )

    phase = htf.Phase(
        phase_name="phase_t2_agg",
        folders=["aggregator", dataset_names[index]],
        profiles=af.CollectionPriorModel(gaussian=htf.profiles.Gaussian),
        settings=htf.SettingsPhase(),
        search=af.DynestyStatic(),
    )

    """Note that we pass the info to the phase when we run it, so that the aggregator can make it accessible."""

    phase.run(dataset=dataset, mask=mask, info=info)

# %%
"""
Checkout the output folder - you should see three new sets of results corresponding to our 3 _Gaussian_ datasets.

Unlike previous tutorials, these folders in the output folder are named after the _Dataset_ and contain the folder
with the phase's name, as opposed to just the phase-name folder.
"""
