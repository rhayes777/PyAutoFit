#%%
"""
Tutorial 4: Source Code
=======================

Up to now, all tutorials have been self contained. That is, the code used to define the model, analysis, run the
non-linear search, load data and plot images were contained in the Jupyter notebooks or Python scritps you were run.

For a real software developmet project, this code woulnd't be contained in one script, but instead make up the
project's source code. In this tutorial, and every tutorial hereafter, we will set up our code as if it were
an actual software project with a clearly defined source code library. This source code can be found in the folder
'/autofit_workspace/howtofit/chapter_1_introduction/tutorial_4_source_code/src'.

Check it out, and first note the directory structure of the source code, which is separated into 5 packages: 'dataset',
'fit', 'model', 'plot' and 'phase'. This cleanly separates the different parts of the code which perform do different
tasks and I recommend your model-fitting project close adopts this structure!

For example, the code which handles the model is completely separate from the code which handles the analysis class.
The model thus never interfaces directly with PyAutoFit, ensuring good code design by removing dependencies between
parts of the code that do not need them! Its the same for the part of the code that stores data ('dataset')
and fits a model to a dataset ('fit') - by keeping them separate its clear which part of the code do what task.

This is a principle aspect of object oriented design and software engineering called 'separation of concerns' and all
templates we provide in the HowToFit series will adhere to it.
"""

# %%
#%matplotlib inline

# %%
from autoconf import conf
import autofit as af
from pyprojroot import here

workspace_path = here()
print("Workspace Path: ", workspace_path)


# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/howtofit/config",
    output_path=f"{workspace_path}/howtofit/output",  # <- This sets up where the non-linear search's outputs go.
)

dataset_path = f"{workspace_path}/howtofit/dataset/chapter_1/gaussian_x1/"

# %%
"""
To begin, checkout the 'data' package, which contains one module, 'dataset.py'. Whereas before we has arrays which
separately contained the data and noise_map, from here on we'll combine them into a 'Dataset' class, which can be 
easily extended if our model-fitting problem has additional data components.

We can use the Dataset's 'from_fits' method to load the dataset for the model-fit we will perform.
"""

from howtofit.chapter_1_introduction.tutorial_4_source_code.src.dataset import (
    dataset as ds,
)

dataset = ds.Dataset.from_fits(
    data_path=f"{dataset_path}/data.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
)

# %%
"""
Previously, we manually specified how to plot the dataset. These plotting functions are now in our source code, in the
'plot' package - check them out now! You'll note we have separate modules for plotting lines (e.g. anything which is 
line, the data, a residual-map, etc.), parts of the dataset or the results of a fit.

You'll notice that the dataset plot functions take instances of the Dataset class. This means we don't have to manually
specify which part of our data we want to pass to the function - just use the function that plots the data!

Lets use a plot function to plot our data.
"""

from howtofit.chapter_1_introduction.tutorial_4_source_code.src.plot import (
    dataset_plots,
)

dataset_plots.data(dataset=dataset)
dataset_plots.noise_map(dataset=dataset)

# %%
"""
Next, look at the 'model' package, which contains the module 'gaussian.py'. This contains the Gaussian class we have
used previous to compose the 1D Gaussian model that we fit.

By packaging all the model components into a single package, this will make it straight forward to add new model
components to the source code.
"""

from howtofit.chapter_1_introduction.tutorial_4_source_code.src.model import (
    gaussian as g,
)

gaussian = g.Gaussian(centre=50.0, intensity=2.0, sigma=20.0)

# %%
"""
Next, lets checkout the 'fit' package which contains the 'fit.py' module. This packages all the fitting methods we 
introduced in tutorial 2 into a single Fit class, making it straight forward to compute the residual-map, chi-squared 
map, log likelihood and so forth. 

The fit plot functions take an instance of this Fit class, making it straight forward to plot the different components
of the fit. Below, I used the Gaussian model above to illustrate how we can easily plot different aspects of a fit. 
"""

# %%
from howtofit.chapter_1_introduction.tutorial_4_source_code.src.plot import fit_plots
from howtofit.chapter_1_introduction.tutorial_4_source_code.src.fit import fit as f

model_data = gaussian.profile_from_xvalues(xvalues=dataset.xvalues)

fit = f.FitDataset(dataset=dataset, model_data=model_data)

fit_plots.residual_map(fit=fit)
fit_plots.normalized_residual_map(fit=fit)
fit_plots.chi_squared_map(fit=fit)

# %%
"""
As we discussed in tutorial 2, the different steps of performing a fit (e.g. computing the residuals, the chi-squared,
log likelihood, and so forth) are pretty much generic tasks performed by any model-fitting problem. 

Thus, you should literally be able to copy and paste the Fit class found in this tutorial (and future tutorials) and 
use them in your modeling software! I have made sure the class works for datasets of higher dimensionality (e.g. 
2D images or 3D datacubes).
"""

# %%
"""
We're finally ready to look at how our source code sets up the non-linear search and model-fit. Whereas before, we 
manually set up the model, analysis and non-linear search, from now on we're going to use PyAutoFits 'phase API' which
uses the 'phase' package, which contains 3 modules: 'phase.py', 'analysis.py' and 'result.py'.

At this point, you should open and inspect (in detail) these 3 source code files. These are the heart of any PyAutoFit 
model fit! 

An over view of each is as follows:

phase.py -> contains the Phase class:

    - Receives the model to be fitted (in this case a single Gaussian).
    - Handles the directory structure of the output (in this example results are output to the folder
      '/output/phase_example/'.
    - Is passed the data when run, which is set up for the analysis.

analysis.py -> contains the Analysis class (is a restructred version of the the previous tutorial's Analysis class):

    - Prepares the dataset for fitting.
    - Fits this dataset with a model instance to compute a log likelihood for every iteration of the non-linear search.

result.py -> contains the Result class:

    - Stores the Samples object containing information on the non-linear search's samples.
    - Has functions to create the model image, residual-map, chi-squared-map and so forth of the maximum log likelihood 
      model etc.
"""

# %%
"""
Perform a non-linear search in PyAutoFit now only requires that we instantiate and run a Phase object. The Phase 
performs the following tasks (which we performed manually in the previous tutorial):

    - Builds the model to be fitted and interfaces it with the non-linear search algorithm.
    - Receives the data to be fitted and prepares it so the model can fit it.
    - Contains the Analysis class that defines the log likelihood function.
    - Returns the results. including the non-linear search's samples and the maximum likelihood fit.

In the previous tutorial, after we composed our model using the _PriorModel_ object we had to manually specify its
priors. However, now we are using a source code, the priors are instead loaded from config files, specifically the
config file found at 'autofit_workspace/config/json_piors/gaussian.json'. If you inspect this file, you'll see the priors
are set up using the same values as the previous tutorila.

It is worth noting that the name of this config file, 'gaussian.json', is not a conincidence. It is named after the
module we imported to create the _PriorModel_, the 'gaussian.py' module. Thus, our the json_config files we use to
set up the default priors of different model components share the name of the module they are in! 

Although we don't in this tutorial, we could of course over write the priors with new priors as we did in the previous
tutorial.

Lets instantiate and run a phase, which reduces the task of performing a model-fit in PyAutoFit to just two lines. 
The results are output to the path 'autofit_workspace/howtofit/chapter_1_introduction/output/phase_t4/emcee', which in 
contrast to the previous tutorial includes the phase name in the path structure.
"""

# %%
from howtofit.chapter_1_introduction.tutorial_4_source_code.src.phase import phase as ph

phase = ph.Phase(
    phase_name="phase_t4", gaussian=af.PriorModel(g.Gaussian), search=af.Emcee()
)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t4/emcee"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

result = phase.run(dataset=dataset)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
The phase returns a Result object, just like the model-fit performed in the previous tutorial did. However, in
'result.py' you'll have noted we extended the Results object to include a property containing an instance of the 
maximum likelihood fit.
"""

# %%
print(result.max_log_likelihood_fit)

# %%
"""
The benefit of writing plot functions that take as input instances of class, specifically in this case the Fit class,
is now clear. We can visualize our results by simply passing this instance to our plots!
"""

# %%
fit_plots.model_data(fit=result.max_log_likelihood_fit)
fit_plots.residual_map(fit=result.max_log_likelihood_fit)
fit_plots.chi_squared_map(fit=result.max_log_likelihood_fit)

# %%
"""
And with that, we have introduced the PyAutoFit phase API alongside an example project, which provides a template on
how to structure model-fitting software. 

All of the remaining tutorials will be provided with a 'src' source code folder, and the functionality these tutorials
describe will be reflected in the comments of the source code. At this point, you should be thinking about how you
might wish to structure your model-fitting software!
"""
