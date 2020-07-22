import autofit as af

# The 'settings.py' module is identical to the previous tutorial.


class PhaseSettings(af.AbstractPhaseSettings):
    def __init__(self, data_trim_left=None, data_trim_right=None):

        super().__init__()

        self.data_trim_left = data_trim_left
        self.data_trim_right = data_trim_right

    @property
    def tag(self):

        # You may well have many more tags which appear here.

        return (
            "settings"  # For every tag you add, you'll add it to this return statement
            + self.data_trim_left_tag
            + self.data_trim_right_tag
            # e.g. + your_own_tag
        )

    # This function generates a string we'll use to 'tag' a phase which uses this setting, thus ensuring results are
    # output to a unique path.

    @property
    def data_trim_left_tag(self):
        """Generate a data trim left tag, to customize phase names based on how much of the dataset is trimmed to 
        its left.
    
        This changes the phase name 'settings' as follows:
    
        data_trim_left = None -> settings
        data_trim_left = 2 -> settings__trim_left_2
        data_trim_left = 10 -> settings__trim_left_10
        """
        if self.data_trim_left is None:
            return ""
        return "__trim_left_" + str(self.data_trim_left)

    @property
    def data_trim_right_tag(self):
        """Generate a data trim right tag, to customize phase names based on how much of the dataset is trimmed to its right.
    
        This changes the phase name 'settings' as follows:
    
        data_trim_right = None -> settings
        data_trim_right = 2 -> settings__trim_right_2
        data_trim_right = 10 -> settings__trim_right_10
        """
        if self.data_trim_right is None:
            return ""
        return "__trim_right_" + str(self.data_trim_right)
