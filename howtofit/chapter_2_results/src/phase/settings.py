import autofit as af

from howtofit.chapter_2_results.src.dataset import dataset as ds

# The `settings.py` module is identical to the previous tutorial.


class SettingsPhase(af.AbstractSettingsPhase):
    def __init__(self, settings_masked_dataset=ds.SettingsMaskedDataset()):
        super().__init__()

        self.settings_masked_dataset = settings_masked_dataset

    @property
    def tag(self):
        """You may well have many more tags which appear here."""

        return "settings" + self.settings_masked_dataset.tag
