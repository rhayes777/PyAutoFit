.. _adding_a_model_component:

Adding a Model Component
========================

Adding a class
--------------

The ``autofit_workspace`` comes ready for fitting 1D ``Gaussian`` and ``Exponential`` profiles, complete with configuration
files, analysis classes and example scripts.

However, once you're familiar with **PyAutoFit**, you will want to add your own model-components, specific to your
model-fitting task. There are a couple of extra steps that come with doing this, associated with configuration files,
that this brief guide explains.

The model-component we are going to add will perform a ``y = mx + c`` linear fit to noisy data drawn from a straight
line. We're only going to focus on the steps necessary to add this new model component, so we'll omit writing an
``Analysis`` class and performing the actual fit itself.

To perform a linear fit, we require a ``LinearFit`` model-component that fits the data with a
line ``y = mx + c`` or equivalently ``y = (gradient * x) + intercept``.

.. code-block:: bash

    class LinearFit:

        def __init__(self, gradient=1.0, intercept=0.0):

            self.gradient = gradient
            self.intercept = intercept

        def profile_from_xvalues(self, xvalues):

            return (self.gradient * xvalues) + self.intercept

As should be clear on by now, the class ``LinearFit`` defines our model-component which has free parameters  ``gradient``
and ``intercept``.

However, if we tried to make this a ``Model`` PyAutoFit would raises an error, e.g.

.. code-block:: bash

    model = af.Model(LinearFit)

The error will read something like ``'KeyError: No prior config found for class LinearFit and path gradient in directories C:\\Users\\Jammy\\Code\\PyAuto\\autofit_workspace\\config\\priors'``.

**PyAutoFit** is informing us that it cannot find prior configuration files for the ``LinearFit`` model-component and that 
they are therefore missing from the folder ``autofit_workspace/config/priors``.


Every model-component must have a ``.json`` config file in the ``autofit_workspace/config/priors`` folder, so 
that **PyAutoFit** knows the default priors to associate with the model-component. If we do not manually override 
priors, these are the priors that will be used by default when a model-fit is performed.

Next, inspect the `TemplateObject.json  <https://github.com/Jammy2211/autofit_workspace/blob/master/config/priors/TemplateObject.json>`_ configuration file in ``autofit_workspace/config/priors``. You should see
the following ``.json`` text:

.. code-block:: bash

    {
        "parameter0": {
            "type": "Uniform",
            "lower_limit": 0.0,
            "upper_limit": 1.0
        },
        "parameter1": {
            "type": "Gaussian",
            "mean": 0.0,
            "sigma": 0.1,
            "lower_limit": "-inf",
            "upper_limit": "inf"
        }
    }

This specifies the default priors on two parameters, named ``parameter0`` and ``parameter1``. The ``type`` is the type of 
prior assumed by **PyAutoFit** by default for its corresponding parameter. 

In the example above: 

- ``parameter0`` is given a ``UniformPrior`` with limits between 0.0 and 1.0. 
- ``parameter1`` a ``GaussianPrior`` with mean 0.0 and sigma 1.0.

The ``lower_limit`` and ``upper_limit`` of a ``GaussianPrior`` define the boundaries of what parameter values are 
physically allowed. If a model-component is given a value outside these limits during model-fitting the model is
instantly resampled and discarded.
 
We can easily adapt this template for our ``LinearFit`` model component. First, copy and paste the `TemplateObject.json  <https://github.com/Jammy2211/autofit_workspace/blob/master/config/priors/TemplateObject.json>`_
file to create a new file called ``LinearFit.json``. 

**PyAutoFit** matches the name of the class to the name of the configuration file, therefore it is a requirement that 
the configuration file is named ``LinearFit.json``.

Next, rename ``parameter0`` to ``gradient``, ``parameter1`` to ``intercept`` and make it so both assume a ``UniformPrior`` 
between -10.0 to 10.0.

The ``.json`` file should read as follows:

.. code-block:: bash

    {
        "gradient": {
            "type": "Uniform",
            "lower_limit": -10.0,
            "upper_limit": 10.0
        },
        "intercept": {
            "type": "Uniform",
            "lower_limit": -10.0,
            "upper_limit": 10.0
        }
    }

We should now be able to make a ``Model`` of the ``LinearFit`` class.

.. code-block:: bash

    model = af.Model(LinearFit)

Adding a Module
---------------

For larger projects, it is not ideal to have to write all the model-component classes in a single Python script, 
especially as we may have many different model components. We instead would prefer them to be in their own dedicated 
Python module.

open the file:

- ``autofit_workspace/scripts/overview/adding_a_model_component/linear_fit.py``  OR
- ``autofit_workspace/notebooks/overview/adding_a_model_component/linear_fit.py``

Here, you will see the ``LinearFit`` class above is contained in the module ``linear_fit.py``. There is also a ``PowerFit`` 
class, fits the function ``y = m (x**p) + c``.

If we import this module and try to make a  ``Model`` of the ``linear_fit.LinearFit`` or ``linear_fit.PowerFit``
classes, we receive the same configuration error as before.

.. code-block:: bash

    import linear_fit
    
    model = af.Model(linear_fit.LinearFit)
    model = af.Model(linear_fit.PowerFit)

This is because if a model-component is contained in a Python module, the prior configuration file must be named after
that ``module`` and structured to contain Python class itself.

Open the file ``autofit_workspace/config/priors/template_module.json``, (https://github.com/Jammy2211/autofit_workspace/blob/master/config/priors/template_module.json)
which reads as follows:

.. code-block:: bash
    
    {
        "ModelComponent0": {
            "parameter0": {
                "type": "Uniform",
                "lower_limit": 0.0,
                "upper_limit": 1.0
            },
            "parameter1": {
                "type": "LogUniform",
                "lower_limit": 1e-06,
                "upper_limit": 1e6
            },
            "parameter2": {
                "type": "Uniform",
                "lower_limit": 0.0,
                "upper_limit": 25.0
            }
        },
        "ModelComponent1": {
            "parameter0": {
                "type": "Uniform",
                "lower_limit": 0.0,
                "upper_limit": 1.0
            },
            "parameter1": {
                "type": "LogUniform",
                "lower_limit": 1e-06,
                "upper_limit": 1e6
            },
            "parameter2": {
                "type": "Uniform",
                "lower_limit": 0.0,
                "upper_limit": 1.0
            }
        }
    }

This looks very similar to ``TemplateObject``, the only differences are:

 - It now contains the model-component class name in the configuration file, e.g. ``ModelComponent0``, ``ModelComponent1``.
 - It includes multiple model-components, whereas ``TemplateObject.json`` corresponded to only one model component.
 
We can again easily adapt this template for our ``linear_fit.py`` module. Copy, paste and rename the ``.json`` file to
``linear_fit.json`` (noting again that **PyAutoFit** matches the module name to the configuration file) and update the
parameters as follows:

.. code-block:: bash
    
    {
        "LinearFit": {
            "gradient": {
                "type": "Uniform",
                "lower_limit": -10.0,
                "upper_limit": 10.0
            },
            "intercept": {
                "type": "Uniform",
                "lower_limit": -10.0,
                "upper_limit": 10.0
            }
        },
        "PowerFit": {
            "gradient": {
                "type": "Uniform",
                "lower_limit": -10.0,
                "upper_limit": 10.0
            },
            "intercept": {
                "type": "Uniform",
                "lower_limit": -10.0,
                "upper_limit": 10.0
            },
            "power": {
                "type": "Uniform",
                "lower_limit": 0.0,
                "upper_limit": 10.0
            }
        }
    }

We are now able to create both the ``linear_fit.LinearFit`` and ``linear_fit.PowerFit`` objects as ``Model``'s.

.. code-block:: bash

    model = af.Model(linear_fit.LinearFit)
    model = af.Model(linear_fit.PowerFit)

Optional Configs
----------------

There are a couple more configuration files you can optionally update, which change how results are output. Open the 
following configuration files:

``autofit_workspace/config/notation/label.ini``
``autofit_workspace/config/notation/label_format.ini``

These configuration files include the following additional settings for our model components:

``label_ini`` -> [label]: 
   This is a short-hand label for each parameter of each model-component used by certain **PyAutoFit** output files.

``label_ini`` -> [subscript]: 
   A subscript for the model-component used by certain **PyAutoFit** output files.

``label_format.ini -> [format]
   The format that the values of a parameter appear in the ``model.results`` file.

For our ``LinearFit`` update the ``label.ini`` config as follows:

.. code-block:: bash

    [label]
    centre=x
    intensity=I
    sigma=sigma
    rate=\lambda
    gradient=m
    intercept=c
    power=p

.. code-block:: bash

    [subscript]
    Gaussian=g
    Exponential=e
    LinearFit=lin
    PowerFit=pow

and ``label_format.ini`` as:

.. code-block:: bash

    [format]
    centre={:.2f}
    intensity={:.2f}
    sigma={:.2f}
    rate={:.2f}
    gradient={:.2f}
    intercept={:.2f}
    power={:.2f}

You should now be able to add your own model-components to your **PyAutoFit** project!