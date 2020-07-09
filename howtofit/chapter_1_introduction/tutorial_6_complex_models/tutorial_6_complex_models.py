# %%
"""
Tutorial 6: Complex Models
==========================

Up to now, we've fitted a very simple model - a single 1D Gaussian with just 3 free parameters. In this tutorial,
we'll look at how PyAutoFit allows us to compose and fit models of arbitrary complexity.

To begin, you should check out the module 'tutorial_6_complex_models/model/profiles.py'. In previous tutorials this
module was called 'gaussian.py' and it contained only the Gaussian class we fitted to data. The module now includes
a second profile, 'Exponential', which like the Gaussian class is a model-component that can be fitted to data.

Up to now, our data has always been generated using a single Gaussian profile. Thus, we have only needed to fit
it with a single Gaussian. In this tutorial, our dataset is now a superpositions of multiple profiles (e.g a
Gaussians + Exponential). The models we compose and fit are therefore composed of multiple profiles, such that when we
generate the model-data we generate it as the sum of all individual profiles in our model.
"""

# %%
#%matplotlib inline

from autoconf import conf
import autofit as af
from howtofit.chapter_1_introduction.tutorial_6_complex_models import (
    src as htf,
)

import numpy as np

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
Lets quickly recap tutorial 1, where using PriorModels we created a Gaussian as a model component and used it to map a
list of parameters to a model instance.
"""

# %%
model = af.PriorModel(htf.profiles.Gaussian)

print("PriorModel Gaussian object: \n")
print(model)

instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3])

print("Model Instance: \n")
print(instance)

print("Instance Parameters \n")
print("x = ", instance.centre)
print("intensity = ", instance.intensity)
print("sigma = ", instance.sigma)

# %%
"""
Defining a model using multiple model components is straight forward in PyAutoFit, using a CollectionPriorModel
object.
"""

# %%
model = af.CollectionPriorModel(
    gaussian=htf.profiles.Gaussian, exponential=htf.profiles.Exponential
)

# %%
"""
A CollectionPriorModel behaves like a _PriorModel_ but contains a collection of model components. For example, it can
create a model instance by mapping a list of parameters, which in this case is 6 (3 for the Gaussian [centre,
intensity, sigma] and 3 for the Exponential [centre, intensity, rate]).
"""

# %%
instance = model.instance_from_vector(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01])

# %%
"""
This instance contains each of the model components we defined above, using the input argument name of the
CollectionoPriorModel to define the attributes in the instance:
"""

# %%
print("Instance Parameters \n")
print("x (Gaussian) = ", instance.gaussian.centre)
print("intensity (Gaussian) = ", instance.gaussian.intensity)
print("sigma (Gaussian) = ", instance.gaussian.sigma)
print("x (Exponential) = ", instance.exponential.centre)
print("intensity (Exponential) = ", instance.exponential.intensity)
print("sigma (Exponential) = ", instance.exponential.rate)

# %%
"""
We can call the components of a CollectionPriorModel whatever we want, and the mapped instance will use those names.
"""

# %%
model_custom_names = af.CollectionPriorModel(
    james=htf.profiles.Gaussian, rich=htf.profiles.Exponential
)

instance = model_custom_names.instance_from_vector(
    vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.01]
)

print("Instance Parameters \n")
print("x (Gaussian) = ", instance.james.centre)
print("intensity (Gaussian) = ", instance.james.intensity)
print("sigma (Gaussian) = ", instance.james.sigma)
print("x (Exponential) = ", instance.rich.centre)
print("intensity (Exponential) = ", instance.rich.intensity)
print("sigma (Exponential) = ", instance.rich.rate)

# %%
"""
Now we can create a model composed of multiple components, lets fit it to a dataset. To do this, we updated this 
tutorial's phase package, spefically its 'Analysis' class such that it creates model data as a super position of all of 
the model's individual profiles. For example, in the model above, the model data is the sum of the Gaussian's 
individual profile and Exponential's individual profile.

Checkout 'phase.py' and 'analysis.py' now, for a description of how this has been implemented.
"""

# %%
"""
Import the simulator module and set up the dataset. This uses a new dataset that is generated as a sum of a Gaussian 
and Exponential profile.
"""

# %%
from howtofit.simulators.chapter_1 import gaussian_x1_exponential_x1

dataset = htf.Dataset(
    data=gaussian_x1_exponential_x1.data, noise_map=gaussian_x1_exponential_x1.noise_map
)

# %%
"""
We again need to create a mask for our data. In this exmample, we'll omit actual masking of the dataset, but still
need to define a mask to pass the 'phase.run' method.
"""

# %%
mask = np.full(fill_value=False, shape=dataset.data.shape)

# %%
"""
Lets now perform the fit using our model which is composed of two profiles. You'll note that the Emcee
dimensionality has increased from N=3 to N=6, given that we are now fitting two profiles each with 3 free parameters.
"""

# %%
phase = htf.Phase(
    phase_name="phase_t6_gaussian_x1_exponential_x1",
    profiles=af.CollectionPriorModel(
        gaussian=htf.profiles.Gaussian, exponential=htf.profiles.Exponential
    ),
    search=af.Emcee(),
)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t5_gaussian_x1_exponential_x1"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Quickly inspect the results of this fit, which you may have noticed takes a bit longer to run than the fits performed
in previous tutorials. This is because the dimensionality of the model we are fitted increased from 3 to 6.

With the CollectionPriorModel, PyAutoFit gives us all the tools we need to compose and fit any model imaginable!
Lets fit a model composed of two Gaussians and and an Exponential, which will have a dimensionality of N=9.
"""

# %%
from howtofit.simulators.chapter_1 import gaussian_x2_exponential_x1

dataset = htf.Dataset(
    data=gaussian_x2_exponential_x1.data, noise_map=gaussian_x2_exponential_x1.noise_map
)

phase = htf.Phase(
    phase_name="phase_t6_gaussian_x2_exponential_x1",
    profiles=af.CollectionPriorModel(
        gaussian_0=htf.profiles.Gaussian,
        gaussian_1=htf.profiles.Gaussian,
        exponential=htf.profiles.Exponential,
    ),
    search=af.Emcee(),
)

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t5_gaussian_x2_exponential_x1"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
We can fully custommize the model that we fit. Lets suppose we have a dataset that consists of three Gaussian line
profiles, but we also know the following information about the dataset:

- All 3 Gaussians are centrally aligned.
- The sigma of one Gaussian is equal to 1.0.

We can edit our CollectionPriorModel to meet these constraints accordingly:
"""

# %%
model = af.CollectionPriorModel(
    gaussian_0=htf.profiles.Gaussian,
    gaussian_1=htf.profiles.Gaussian,
    gaussian_2=htf.profiles.Gaussian,
)

# %%
"""
This aligns the centres of the 3 Gaussians, reducing the dimensionality of the model from N=9 to N=7
"""

# %%
model.gaussian_0.centre = model.gaussian_1.centre
model.gaussian_1.centre = model.gaussian_2.centre

# %%
"""
This fixes the sigma value of one Gaussian to 1.0, further reducing the dimensionality from N=7 to N=6.
"""

# %%
model.gaussian_0.sigma = 1.0

# %%
"""
We can now fit this model using a phase as per usual.
"""

# %%
from howtofit.simulators.chapter_1 import gaussian_x3

dataset = htf.Dataset(data=gaussian_x3.data, noise_map=gaussian_x3.noise_map)

phase = htf.Phase(phase_name="phase_t6_gaussian_x3", profiles=model, search=af.Emcee())

print(
    "Emcee has begun running - checkout the autofit_workspace/howtofit/chapter_1_introduction/output/phase_t5_gaussian_x3"
    "folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

phase.run(dataset=dataset, mask=mask)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
And with that, we are complete. In this tutorial, we learned how to compose complex models in PyAutoFit and adjust our
'phase.py' and 'analyis.py' modules to fit them. To end, you should think again in more detail about your model
fitting problem:

Are there many different model components you may wish to define and fit?
Is your model-data the super position of many different model components, like the profiles in this tutorial?

In this tutorial, all components of our model did the same thing - represent a 'line' of data. In your model, you may
have model components that represent different parts of your model, which need to be combined in more complicated ways
in order to create your model-fit. In such circumstances, the 'fit' method in your 'Analysis' class may be significantly
more complex than the example shown in this tutorial. Nevertheless, you now have all the tools you need to define,
compose and fit very complex models - there isn't much left for you to learn on your journey through PyAutoFit!
"""
