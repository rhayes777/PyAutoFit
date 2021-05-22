.. _topological:

Topological Models
------------------

Using hierarchies of Python classes **PyAutoFit** can construct **topological models**, which consist of many
model-components (composed via the ``Model`` and ``Collection`` objects) that are linked together to form one
over-arching model.

Topological models are a fairly abstract concept, and so to describe them we are going to introduce a real-world
model-fitting example. We will use the an example from Astronomy, fitting images of gravitationally lensed galaxies.
This is the science case that sparked the development of **PyAutoFit** as a spin off of our astronomy software
`PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_.

The schematic below depicts a strong gravitational lens:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAuto/master/docs/overview/images/lensing/schematic.jpg
  :width: 600
  :alt: Alternative text

**Credit: F. Courbin, S. G. Djorgovski, G. Meylan, et al., Caltech / EPFL / WMKO**
https://www.astro.caltech.edu/~george/qsolens/

A strong gravitational lens is a system consisting of multiple galaxy's down the light-of-sight to earth. To model
a strong lens, we effectively ray-trace the traversal of light throughout the Universe so as to fit it to imaging
data of a strong lens. The amount light is deflected by is defined by the distances between each galaxy, caleld their
redshift.

We therefore need a model which contains separate model-components for every galaxy, and where each galaxy contains
separate model-components describing its light and mass:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoFit/master/docs/overview/images/lens_model.png
  :width: 600
  :alt: Alternative text

The image above shows that we need a model consisting of individual model-components for:

 1) The lens galaxy's light and mass.
 2) The source galaxy's light.

However, we also need each galaxy to be a model-component and for it to contain an additional parameter, its redshift.
The galaxies can then finally be combined into an overall model for the strong lens system.

To model the light of a galaxy, we can define a ``LightProfile`` as a Python class, which behaves in the same way as
the ``Gaussian`` used throughout over tutorials:

.. code-block:: bash

    class LightDeVaucouleurs:

        def __init__(
            self,
            centre: typing.Tuple[float, float] = (0.0, 0.0),
            intensity: float = 0.1,
            radius: float = 0.6,
        ):
            """
            The De Vaucouleurs light profile often used in Astronomy to represent the bulge of galaxies.

            Parameters
            ----------
            centre
                The (y,x) coordinates of the profile centre.
            intensity
                Overall intensity normalisation of the light profile.
            radius
                The circular radius containing half the light of this profile.
            """

            self.centre = centre
            self.intensity = intensity
            self.effective_radius = effective_radius

        def image_from_grid(self, grid: np.ndarray) -> np.ndarray:
            """This function creates an image of the light profile, which is used in strong lens model-fitting"""
            ...

We have omitted the code that creates the image from the light profile as we want to focus purely on topological model
composition with **PyAutoFit**. Checkout the example project on the autofit_workspace for the complete code.

We also define a ``MassProfile``:

.. code-block:: bash

    class MassIsothermal:
        def __init__(
            self,
            centre: typing.Tuple[float, float] = (0.0, 0.0),
            mass: float = 1.0,
        ):
            """
            The isothermal mass distribution often used in Astronomy to represent the combined mass of stars
            and dark matter in galaxies.

            Parameters
            ----------
            centre
                The (y,x) coordinates of the profile centre.
            mass
                The mass normalization of the profile, which is the Einstein radius in arc-seconds.
            """

            self.centre = centre
            self.mass = mass

        def deflections_from_grid(self, grid: np.ndarray) -> np.ndarray:
            """This function describes the deflection of light due to the mass, which is used in strong lens model-fitting"""
            ...

We now need to define a ``Galaxy`` object, which contains instances of light and mass profiles and its redshift (e.g.
distance from Earth):

.. code-block:: bash

    class Galaxy:

        def __init__(
            self,
            redshift: float,
            light_profiles: Optional[List] = None,
            mass_profiles: Optional[List] = None,
        ):
            """
            A galaxy, which contains light and mass profiles at a specified redshift.

            Parameters
            ----------
            redshift
                The redshift of the galaxy.
            light_profiles
                A list of the galaxy's light profiles.
            mass_profiles
                A list of the galaxy's mass profiles.
            """

            self.redshift = redshift
            self.light_profiles = light_profiles
            self.mass_profiles = mass_profiles

        def image_from_grid(self, grid: np.ndarray) -> np.ndarray:
            """Returns the image of all light profiles."""
            ...

        def deflections_from_grid(self, grid: np.ndarray) -> np.ndarray:
            """Returns the deflection angles of all mass profiles."""
            ...

If we were not composing a model, the code below shows how one would create an instance of the lens and source galaxies
and use them to ray-trace light.

.. code-block:: bash

    light = LightDeVaucouleurs(centre=(0.0, 0.0), intensity=10.0, radius=2.0)
    mass = MassIsothermal(centre=(0.0, 0.0), mass=0.5)

    lens = Galaxy(redshift=0.5, light_profiles=[light], mass_profiles=[mass])


    light = LightDeVaucouleurs(centre=(1.0, 0.5), intensity=2.0, radius=5.0)

    source = Galaxy(redshift=1.0, light_profiles=[light])

The code above uses a hierarchicy of class instances. That is, instances of the ``LightDeVaucouleurs``
and ``MassIsothermal`` classes are created and used to create an instance of the ``Galaxy`` class.

We can compose a topological model using this same hierarchy of classes, using the ``Model`` and ``Collection`` objects
that were introduced previously. Lets first create a model of the lens galaxy:

.. code-block:: bash

    lens = af.Model(Galaxy, light_profiles=[LightDeVaucouleurs], mass_profiles=[MassIsothermal])

