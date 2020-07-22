from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.dataset import (
    dataset as ds,
)

# The 'meta_dataset.py' module is unchanged from the previous tutorial.


class MetaDataset:
    def __init__(self, settings):
        """An intermediate meta-dataset class which is used to create the masked dataset, but with phase-dependent
        settings applied that augment the data. In this example, the left and right edges of the data and mask can be
        trimmed.

        Parameters
        ----------
        settings : PhaseSettings
            The collection of settings of the phase used to augment the data that is fitted and tag the output path.
        """

        self.settings = settings

    def masked_dataset_from_dataset_and_mask(self, dataset, mask):
        """Create a masked dataset from the input dataset and a mask.

        The dataset will have its left and / or right edge trimmed and removed from the model-fit, if the phase
        settings are input to do this.

        Parameters
        ----------
        dataset : dataset.Dataset
            The dataset that is masked, trimmed and fitted.
        mask : ndarray
            The mask applied to the dataset, which is also trimmed.
        """

        masked_dataset = ds.MaskedDataset(dataset=dataset, mask=mask)

        if self.settings.data_trim_left is not None:
            masked_dataset = masked_dataset.with_left_trimmed(
                data_trim_left=self.settings.data_trim_left
            )

        if self.settings.data_trim_right is not None:
            masked_dataset = masked_dataset.with_right_trimmed(
                data_trim_right=self.settings.data_trim_right
            )

        return masked_dataset
