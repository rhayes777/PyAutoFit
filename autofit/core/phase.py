from autofit.core import non_linear


class AbstractPhase(object):

    def __init__(self, optimizer_class=non_linear.MultiNest, phase_name=None,
                 auto_link_priors=False):
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
        self.optimizer = optimizer_class(name=phase_name)
        self.phase_name = phase_name
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

        def __init__(self, phase_name, previous_results=None):
            """
            An lensing object

            Parameters
            ----------
            phase_name: str
                The name of the phase to which this analysis belongs
            previous_results: ResultsCollection
                The results of all previous phases
            """

            self.previous_results = previous_results
            self.phase_name = phase_name

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

        def tracer_for_instance(self, instance):
            raise NotImplementedError()

        def padded_tracer_for_instance(self, instance):
            raise NotImplementedError()

        def fit_for_tracers(self, tracer, padded_tracer):
            raise NotImplementedError()

        def fast_likelihood_for_tracer(self, tracer):
            raise NotImplementedError()

    class Result(non_linear.Result):

        def __init__(self, constant, likelihood, variable):
            """
            The result of a phase
            """
            super(AbstractPhase.Result, self).__init__(constant, likelihood, variable)
