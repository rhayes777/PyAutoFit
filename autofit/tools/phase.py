import re

from autofit.tools import path_util
from autofit.optimize import grid_search
from autofit.optimize import non_linear


class ResultsCollection(list):
    def __init__(self, results):
        super().__init__(results)

    @property
    def last(self):
        if len(self) > 0:
            return self[-1]
        return None

    @property
    def first(self):
        if len(self) > 0:
            return self[0]
        return None


def make_name(cls):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class AbstractPhase(object):

    def __init__(self, phase_name, phase_folders=None, optimizer_class=non_linear.MultiNest, auto_link_priors=False):
        """
        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to fit_normal models and image
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """
        self.phase_folders = phase_folders
        if phase_folders is None:
            self.phase_path = ''
        else:
            self.phase_path = path_util.path_from_folder_names(folder_names=phase_folders)
        self.phase_name = phase_name
        self.optimizer = optimizer_class(phase_folders=self.phase_folders, phase_name=self.phase_name)
        self.auto_link_priors = auto_link_priors

    @property
    def constant(self):
        """
        Convenience method

        Returns
        -------
        ModelInstance
            A model instance comprising all the constant objects in this lensing
        """
        return self.optimizer.constant

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this lensing
        """
        return self.optimizer.variable

    def run_analysis(self, analysis):
        return self.optimizer.fit(analysis)

    @property
    def path(self):
        return self.optimizer.path

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def pass_priors(self, previous_results):
        """
        Perform any prior or constant passing. This could involve setting model attributes equal to priors or constants
        from a previous phase.

        Parameters
        ----------
        previous_results: ResultsCollection
            The result of the previous phase
        """
        pass

    # noinspection PyAbstractClass
    class Analysis(non_linear.Analysis):

        def __init__(self, previous_results=None):
            """
            An lensing object

            Parameters
            ----------
            previous_results: ResultsCollection
                The results of all previous phases
            """

            self.previous_results = previous_results

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

    def make_result(self, result, analysis):
        raise NotImplementedError()


def as_grid_search(phase_class):
    class GridSearchExtension(phase_class):
        def __init__(self, *args, phase_name, phase_folders=None, number_of_steps=10, optimizer_class=non_linear.MultiNest, **kwargs):
            super().__init__(*args, phase_name=phase_name, phase_folders=phase_folders, optimizer_class=optimizer_class,
                             **kwargs)
            self.optimizer = grid_search.GridSearch(phase_name=phase_name, phase_folders=phase_folders,
                                                    number_of_steps=number_of_steps, optimizer_class=optimizer_class,
                                                    model_mapper=self.variable, constant=self.constant)

        def run_analysis(self, analysis):
            return self.optimizer.fit(analysis, self.grid_priors)

        # noinspection PyMethodMayBeStatic,PyUnusedLocal
        def make_result(self, result, analysis):
            return result

        @property
        def grid_priors(self):
            raise NotImplementedError(
                "The grid priors property must be implemented to provide a list of priors to be grid searched")

    return GridSearchExtension
