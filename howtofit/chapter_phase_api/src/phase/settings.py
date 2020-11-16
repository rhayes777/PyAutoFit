import autofit as af

from src.dataset import dataset as ds

"""
This module create tags for phases settings that customize the analysis. We tag phases for two reasons:

    1) Tags describes the phase settings, making it explicit what analysis was used to create the results.

    2) Tags create unique output paths, ensuring that if you run multiple phases on the same data with different settings
       each `NonLinearSearch` (e.g. Emcee) won't inadvertently use results generated via a different analysis method.
       
The settings of a `SettingsPhase` use the settings objects of individual parts of the code. For example, below, it uses
as input a `SettingsDataset` object, which in the module `dataset/dataset.py` can be seen contains the inputs and
tags required for trimming data.

You may be surprised to see us using classes like `SettingsDataset` and `SettingsPhase`, as the use of a class
may appear somewhat unecessary. We will discuss the use of classes and structure of the code at the end of the 
tutorial script.
"""


class SettingsPhase(af.AbstractSettingsPhase):
    def __init__(self, settings_dataset=ds.SettingsDataset()):

        super().__init__()

        self.settings_dataset = settings_dataset

    """
    This function generates a string we'll use to `tag` a phase which uses this setting, thus ensuring results are
    output to a unique path.
    
    It uses the tag of the `SettingsDataset` to do this.
    """

    @property
    def tag(self) -> str:

        """You may well have many more tags which appear here."""

        return f"settings{self.settings_dataset.tag}"
