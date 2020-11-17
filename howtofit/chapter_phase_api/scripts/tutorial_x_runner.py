# %%
"""
__Pipelines__
"""

# %%
#%matplotlib inline

# %%
from autoconf import conf
import numpy as np

from howtofit.chapter_phase_api_non_linear_searches.src.dataset import (
    dataset as ds,
)

# %%
"""
You need to change the path below to the chapter 2 directory so we can load the dataset
"""

# %%
chapter_path = "/Users/Jammy/Code/PyAuto/autofit_workspace/howtofit/chapter_phase_api_non_linear_searches"

# %%
"""
Setup the configs as we did in the previous tutorial, as well as the output folder for our `NonLinearSearch`.
"""

# %%
conf.instance = conf.Config(
    config_path=f"howtofit/config", output_path=f"howtofit/output/chapter_phase_api"
)

# %%
"""
Lets load the dataset, create a mask and perform the fit.
"""

# %%
dataset_path = f"{chapter_path}/dataset/gaussian_x2_split/"

dataset = ds.Dataset(data=data, noise_map=noise_map)

# %%
"""
Lets use a plot function to plot our data.

Note how - describe its x2 peaks.
"""

from howtofit.chapter_phase_api_non_linear_searches.src.plot import (
    dataset_plots,
)

dataset_plots.data(dataset=dataset)

mask = np.full(fill_value=False, shape=dataset.data.shape)

print(
    "Pipeline has begun running - checkout the autofit_workspace/howtofit/chapter_phase_api_non_linear_searches/output/pipeline__x2_gaussians"
    " folder for live output of the results."
    "This Jupyter notebook cell with progress once the Pipeline has completed - this could take a few minutes!"
)

from howtofit.chapter_phase_api_non_linear_searches import (
    tutoial_x_pipeline,
)

pipeline = tutoial_x_pipeline.make_pipeline()

pipeline.run(dataset=dataset, mask=mask)
