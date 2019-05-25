from autofit.optimize import non_linear
import math

class MockNonLinearOptimizer(non_linear.NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, model_mapper=None,
                 most_probable=None, most_likely=None):

        super(MockNonLinearOptimizer, self).__init__(phase_name=phase_name, phase_tag=phase_tag,
                                                     phase_folders=phase_folders, model_mapper=model_mapper)

        self.most_probable = most_probable
        self.most_likely = most_likely

    @property
    def most_probable_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        return self.most_probable

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.most_likely

    def model_parameters_at_sigma_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))
        densities_1d = list(map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names))
        return list(map(lambda p: p.getLimits(limit), densities_1d))

    @property
    def pdf(self):
        import getdist
        return getdist.mcsamples.loadMCSamples(self.backup_path + '/nlo')

    def model_parameters_at_upper_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[1], self.model_parameters_at_sigma_limit(sigma_limit)))

    def model_parameters_at_lower_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[0], self.model_parameters_at_sigma_limit(sigma_limit)))


class MockClassNLOx4(object):

    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockClassNLOx5(object):

    def __init__(self, one=1, two=2, three=3, four=4, five=5):
        self.one = one
        self.two = two
        self.three = three
        self.four = four
        self.five = five


class MockClassNLOx6(object):

    def __init__(self, one=(1, 2), two=(3, 4), three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockAnalysis(object):
    def __init__(self):
        self.kwargs = None
        self.instance = None
        self.visualise_instance = None

    def fit(self, instance):
        self.instance = instance
        return 1.

    # noinspection PyUnusedLocal
    def visualize(self, instance, *args, **kwargs):
        self.visualise_instance = instance