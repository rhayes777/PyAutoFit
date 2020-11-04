# %%
"""
Tutorial 2: Dataset Sample
==========================

In this tutorial, we'll fit multiple `Dataset`s with the same phase, producing multiple sets of results on our
hard-disk. In the following tutorials, we then use these results and the `Aggregator` to load the results into
our Jupyter notebook to interpret, inspect and plot the output results.

we'll fit 3 different `Dataset`s, each with a single `Gaussian` model.
"""

# %%
#%matplotlib inline

from pyprojroot import here
workspace_path = str(here())
%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from howtofit.chapter_2_results import src as htf
import numpy as np

# %%
"""
Here, for each `Dataset` we are going to set up the correct path, load it, create its `mask` and fit it using a `Phase`.

We want our results to be in a folder specific to the `Dataset`. we'll use the `Dataset`'s name string to do this. Lets
create a list of all 3 of our `Dataset` names.

we'll also pass these names to the `Dataset` when we create it, the name will be used by the `Aggregator` to name the 
file the data is stored. More importantly, the name will be accessible to the `Aggregator`, and we will use it to label 
figures we make via the `Aggregator`.
"""

# %%
dataset_names = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

# %%
"""
We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the data of observation) that isn't part of the model-fit is made accessible to 
the `Aggregator`. For example, below we write info on the `Dataset`'s data of observation and exposure time.
"""

# %%
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

# %%
"""
This for loop runs over every `Dataset`, checkout the comments below for how we set up the path structure.
"""

# %%
for dataset_name in dataset_names:

    """The code below creates the `Dataset` and `mask` as per usual."""

    dataset_path = f"dataset/howtofit/chapter_2/{dataset_name}"

    data = af.util.numpy_array_from_json(file_path=f"{dataset_path}/data.json")
    noise_map = af.util.numpy_array_from_json(
        file_path=f"{dataset_path}/noise_map.json"
    )
    dataset = htf.Dataset(data=data, noise_map=noise_map)

    mask = np.full(fill_value=False, shape=dataset.data.shape)

    """
    Here, we create a `Phase` as normal. However, we also include an input parameter `path_prefix`. This defines the 
    folders that the `Phase` outputs results to before the `name`. For example, if a `Phase` outputs to the path:

        `/path/to/autofit_workspace/output/name/`

    A `Phase` with the `path_prefix='phase_folder' edits this path to:

        `/path/to/autofit_workspace/output/phase_folder/name/`

    You can input multiple folders, for example `path_prefix='folder_0/folder_1' would create the path:

        `/path/to/autofit_workspace/output/folder_0/folder_1/name/`

    Below, we use the `data_name`, so our results go in a folder specific to the `Dataset`, e.g:

        `/path/to/autofit_workspace/output/gaussian_x1_0/phase_t2/`
    """

    print(
        f"Emcee has begun running - checkout the "
        f"autofit_workspace/output/howtofit/chapter_2/aggregator/{dataset_name}/phase_t2_agg folder for live "
        f"output of the results. This Jupyter notebook cell with progress once Emcee has completed - this could take a "
        f"few minutes!"
    )

    phase = htf.Phase(
        search=af.DynestyStatic(
            path_prefix=f"howtofit/chapter_2/aggregator/{dataset_name}",
            name="phase_t2_agg",
        ),
        profiles=af.CollectionPriorModel(gaussian=htf.profiles.Gaussian),
        settings=htf.SettingsPhase(),
    )

    """Note that we pass the info to the `phase` when we run it, so that the `Aggregator` can make it accessible."""

    phase.run(dataset=dataset, mask=mask, info=info)

# %%
"""
Checkout the output folder - you should see three new sets of results corresponding to our 3 `Gaussian` datasets.

Unlike previous tutorials, these folders in the output folder are named after the `Dataset` and contain the folder
with the phase`s name, as opposed to just the phase-name folder.
"""
