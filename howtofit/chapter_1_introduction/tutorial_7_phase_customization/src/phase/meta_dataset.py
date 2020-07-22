from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.dataset import (
    dataset as ds,
)

"""
In this tutorial, we run phases where the dataset we input into the phase is altered before the model-fitting
procedure is run. The dataset is trimmed by an input number of pixels to the left and / or right.

The 'meta_dataset.py' module is the module in PyAutoFit which handles the creation of these new datasets. If we want
to have the option of a phase editing the data-set, the parameters which control this (e.g. 'data_trim_left')
are stored here and then used when the 'phase.run' method is called.

Your model-fitting problem may not require the meta_dataset.py module. If so that is fine, and you can revert to the
templates of the previous tutorials which do not use one. It really depends on the nature of your problem.
"""


class MetaDataset:

    """
    The data_trim_left and data_trim_right are passed to the phase when it is set up and stored in an
    instance of the 'MetaDataset' class.
    """

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

    """
    The masked dataset that is fitted by an analysis is created by the MetaDataset class using the method below.

    If the MetaDataset's data trim attributes are not None, they are used to trim the masked-dataset before it is
    fitted.
    """

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
