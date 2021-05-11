from autofit import exc


class Result:
    """
    @DynamicAttrs
    """

    def __init__(self, samples, model, search=None):
        """
        The result of an optimization.

        Parameters
        ----------
        model
            The model mapper from the stage that produced this result
        """

        self.samples = samples
        self.search = search

        self._model = model
        self.__model = None

        self._instance = (
            samples.max_log_likelihood_instance if samples is not None else None
        )

    @property
    def log_likelihood(self):
        return max(self.samples.log_likelihood_list)

    @property
    def instance(self):
        return self._instance

    @property
    def max_log_likelihood_instance(self):
        return self._instance

    @property
    def model(self):
        if self.__model is None:
            tuples = self.samples.gaussian_priors_at_sigma(
                sigma=self.search.prior_passer.sigma
            )
            self.__model = self._model.mapper_from_gaussian_tuples(
                tuples,
                use_errors=self.search.prior_passer.use_errors,
                use_widths=self.search.prior_passer.use_widths
            )
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __str__(self):
        return "Analysis Result:\n{}".format(
            "\n".join(
                ["{}: {}".format(key, value) for key, value in self.__dict__.items()]
            )
        )

    def model_absolute(self, a: float) -> "mm.ModelMapper":
        """
        Parameters
        ----------
        a
            The absolute width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined absolute
        width.
        """
        return self.model.mapper_from_gaussian_tuples(
            self.samples.gaussian_priors_at_sigma(sigma=self.search.prior_passer.sigma), a=a
        )

    def model_relative(self, r: float) -> "mm.ModelMapper":
        """
        Parameters
        ----------
        r
            The relative width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined relative
        width.
        """
        return self.model.mapper_from_gaussian_tuples(
            self.samples.gaussian_priors_at_sigma(sigma=self.search.prior_passer.sigma), r=r
        )


class ResultsCollection:
    def __init__(self, result_list=None):
        """
        A collection of results from previous searchs. Results can be obtained using an index or the name of the search
        from whence they came.
        """
        self.__result_list = []
        self.__result_dict = {}

        if result_list is not None:
            for result in result_list:
                self.add(name="", result=result)

    def copy(self):
        collection = ResultsCollection()
        collection.__result_dict = self.__result_dict
        collection.__result_list = self.__result_list
        return collection

    @property
    def reversed(self):
        return reversed(self.__result_list)

    @property
    def last(self):
        """
        The result of the last search
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    @property
    def first(self):
        """
        The result of the first search
        """
        if len(self.__result_list) > 0:
            return self.__result_list[0]
        return None

    def add(self, name, result):
        """
        Add the result of a search.

        Parameters
        ----------
        name: str
            The name of the search
        result
            The result of that search
        """
        try:
            self.__result_list[self.__result_list.index(result)] = result
        except ValueError:
            self.__result_list.append(result)
        self.__result_dict[name] = result

    def __getitem__(self, item):
        """
        Get the result of a previous search by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous search
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)

    def from_name(self, name):
        """
        Returns the result of a previous search by its name

        Parameters
        ----------
        name: str
            The name of a previous search

        Returns
        -------
        result: Result
            The result of that search

        Raises
        ------
        exc.PipelineException
            If no search with the expected result is found
        """
        try:
            return self.__result_dict[name]
        except KeyError:
            raise exc.PipelineException(
                "No previous search named {} found in results ({})".format(
                    name, ", ".join(self.__result_dict.keys())
                )
            )

    def __contains__(self, item):
        return item in self.__result_dict
