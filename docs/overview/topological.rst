.. _topological:

Topological Models
------------------

Using hierarchies of Python classes **PyAutoFit** can construct **topological models**, which consist of many
model-components (composed via the ``Model`` and ``Collection`` objects) that are linked together to form one
over-arching model.

Topological models are a fairly abstract concept, and so to describe them we are going to introduce a real-world
model-fitting example. We will use an example from Astronomy; fitting images of gravitationally lensed galaxies.
This is the science case that sparked the development of **PyAutoFit** as a spin off of our astronomy software
`PyAutoLens <https://github.com/Jammy2211/PyAutoLens>`_.

The schematic below depicts a strong gravitational lens:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAuto/master/docs/overview/images/lensing/schematic.jpg
  :width: 600
  :alt: Alternative text

**Credit: F. Courbin, S. G. Djorgovski, G. Meylan, et al., Caltech / EPFL / WMKO**
https://www.astro.caltech.edu/~george/qsolens/

A strong gravitational lens is a system consisting of multiple galaxy's down the light-of-sight to earth. To model
a strong lens, we ray-trace the traversal of light throughout the Universe so as to fit it to imaging data of a strong
lens. The amount light is deflected by is defined by the distances between each galaxy, which is called their redshift.

We therefore need a model which contains separate model-components for every galaxy, and where each galaxy contains
separate model-components describing its light and mass:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoFit/master/docs/overview/images/lens_model.png
  :width: 600
  :alt: Alternative text

The image above shows that we need a model consisting of individual model-components for:

 1) The lens galaxy's light and mass.
 2) The source galaxy's light.

However, we also need each galaxy to be a model-component and for it to contain an additional parameter,
its ``redshift``. The galaxies can then finally be combined into an overall model for the strong lens system.

To model the light of a galaxy, we define a ``LightProfile`` as a Python class, which behaves in the same way as
the ``Gaussian`` used in other **PyAutoFit** tutorials:

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
composition with **PyAutoFit**.

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

We have again omitted the code which computes how this mass profile deflects the path of light.

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

This code creates instances of the ``LightDeVaucouleurs`` and ``MassIsothermal`` classes uses them to create an
instance of the ``Galaxy`` class. It creates a ``Galaxy`` using a **hierarchy of Python classes**.

We can compose a topological model using this same hierarchy of classes, using the ``Model`` and ``Collection`` objects
that were introduced previously. Lets first create a model of the lens galaxy:

.. code-block:: bash

    lens = af.Model(
        Galaxy,
        redshift=0.5,
        light_profiles=[LightDeVaucouleurs],
        mass_profiles=[MassIsothermal]
    )

Lets consider what the code above is doing:

 1) We use a `Model` to create the overall model component. The `cls` input is the `Galaxy` class, therefore the
 overall model that is created is a `Galaxy`.

 2) **PyAutoFit** next inspects whether the key word argument inputs to the `Model` match any of the `__init__`
constructor arguments of the `Galaxy` class. This determine if these inputs are to be composed as model sub-components.

 3) **PyAutoFit** matches the `light_profiles` and  `mass_profiles` inputs, noting they are passed as separate lists
 containing the `LightDeVaucouleurs` and `MassIsothermal` class. They are created as sub-components of the overall
 `Galaxy` model.

 4) It also matches the `redshift` input, making it a fixed value of 0.5 for the model and not treating it as a
 free parameter.

We can confirm this by printing the `prior_count` of the lens, and noting it is 11 (6 parameters for
the `LightDeVaucouleurs` and 5 for the `MassIsothermal`).

.. code-block:: bash

    print(lens.prior_count)
    print(lens.light_profiles[0].prior_count)
    print(lens.mass_profiles[0].prior_count)

The `lens` behaves exactly like the model-components we are used to previously. For example, we can unpack its
individual parameters to customize the model, where below we:

 1) Align the light profile centre and mass profile centre.
 2) Customize the prior on the light profile `axis_ratio`.
 3) Fix the `axis_ratio` of the mass profile to 0.8.

.. code-block:: bash

    lens.light_profiles[0].centre = lens.mass_profiles[0].centre
    lens.light_profiles[0].axis_ratio = af.UniformPrior(lower_limit=0.7, upper_limit=0.9)
    lens.mass_profiles[0].axis_ratio = 0.8

We can now create a model of our source galaxy using the same API.

.. code-block:: bash

    source = af.Model(
        astro.Galaxy,
        redshift=1.0,
        light_profiles=[af.Model(astro.lp.LightExponential)]
    )

We can now create our overall strong lens model, using a `Collection` in the same way we have seen previously.

.. code-block:: bash

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

The model contains both galaxies in the strong lens, alongside all of their light and mass profiles.

For every iteration of the non-linear search **PyAutoFit** generates an instance of this model, where all of the
`LightDeVaucouleurs`, `MassIsothermal` and `Galaxy` parameters of the are determined via their priors.

An example instance is show below:

.. code-block:: bash

    print("Strong Lens Model Instance:")
    print("Lens Galaxy = ", instance.galaxies.lens)
    print("Lens Galaxy Bulge = ", instance.galaxies.lens.light_profiles)
    print("Lens Galaxy Bulge Centre = ", instance.galaxies.lens.light_profiles[0].centre)
    print("Lens Galaxy Mass Centre = ", instance.galaxies.lens.mass_profiles[0].centre)
    print("Source Galaxy = ", instance.galaxies.source)

This model can therefore be used in a **PyAutoFit** `Analysis` class and `log_likelihood_function`.

**Extensibility:**

This example project highlights how topological models can make certain model-fitting problem fully extensible. For
example:

 1) A `Galaxy` class can be created using any combination of light and mass profiles, because it implements their
 `image_from_grid` and `deflections_from_grid` methods as the sum of individual profiles.

 2) The overall strong lens model can contain any number of `Galaxy`'s, as these methods and their redshifts are used
 to implement the lensing calculations in the `Analysis` class and `log_likelihood_function`.

Thus, for problems of this nature, we can design and write code in a way that fully utilizes **PyAutoFit**'s topological
modeling features to compose and fits models of arbritrary complexity and dimensionality.

To illustrate this further, consider the following dataset which is called a *strong lens galaxy cluster*:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoFit/master/docs/overview/images/cluster_example.png
  :width: 600
  :alt: Alternative text

For this strong lens, there are many tens of strong lens galaxies as well as multiple background source galaxies.
However, despite it being a significantly more complex system than the single-galaxy strong lens we modeled above,
our use of topological models ensures that we can model such datasets without any additional code development, for
example:

.. code-block:: bash

    lens_0 = af.Model(
        Galaxy,
        redshift=0.5,
        light_profiles=[LightDeVaucouleurs],
        mass_profiles=[MassIsothermal]
    )

    lens_1 = af.Model(
        Galaxy,
        redshift=0.5,
        light_profiles=[LightDeVaucouleurs],
        mass_profiles=[MassIsothermal]
    )

    source_0 = af.Model(
        astro.Galaxy,
        redshift=1.0,
        light_profiles=[af.Model(astro.lp.LightExponential)]
    )

    # ... repeat for desired model complexity ...

    model = af.Collection(
        galaxies=af.Collection(
            lens_0=lens_0,
            lens_1=lens_1,
            source_0=source_0,
            # ... repeat for desired model complexity ...
        )
    )

**PyAutoFit** therefore gives us full control over the composition and customization of high dimensional topological
models.