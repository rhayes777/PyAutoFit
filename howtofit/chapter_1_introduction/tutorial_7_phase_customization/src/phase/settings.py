import autofit as af

from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.dataset import (
    dataset as ds,
)

"""
This module create tags for phases settings that customize the analysis. We tag phases for two reasons:

    1) Tags describes the phase settings, making it explicit what analysis was used to create the results.

    2) Tags create unique output paths, ensuring that if you run multiple phases on the same data with different settings
       each `NonLinearSearch` (e.g. Emcee) won't inadvertently use results generated via a different analysis method.
       
The settings of a `SettingsPhase` use the settings objects of individual parts of the code. For example, below, it uses
as input a `SettingsMaskedDataset` object, which in the module `dataset/dataset.py` can be seen contains the inputs and
tags required for trimming data.

You may be surprised to see us using classes like `SettingsMaskedDataset` and `SettingsPhase`, as the use of a class
may appear somewhat uncessary. We will discuss the use of classes and structure of the code at the end of the tutorial 
script.
"""


class SettingsPhase(af.AbstractSettingsPhase):
    def __init__(self, settings_masked_dataset=ds.SettingsMaskedDataset()):

        super().__init__()

        self.settings_masked_dataset = settings_masked_dataset

    """
    This function generates a string we'll use to `tag` a phase which uses this setting, thus ensuring results are
    output to a unique path.
    
    It uses the tag of the `SettingsMaskedDataset` to do this.
    """

    @property
    def tag(self):

        """You may well have many more tags which appear here."""

        return "settings" + self.settings_masked_dataset.tag
