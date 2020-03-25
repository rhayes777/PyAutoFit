class ArithmeticMixin:
    def __add__(self, other):
        from autofit.mapper.prior_model.compound import SumPrior
        return SumPrior(
            self, other
        )

    def __radd__(self, other):
        from autofit.mapper.prior_model.compound import SumPrior
        return SumPrior(
            other, self
        )

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        from autofit.mapper.prior_model.compound import NegativePrior
        return NegativePrior(
            self
        )

    def __mul__(self, other):
        from autofit.mapper.prior_model.compound import MultiplePrior
        return MultiplePrior(
            self, other
        )

    def __rmul__(self, other):
        from autofit.mapper.prior_model.compound import MultiplePrior
        return MultiplePrior(
            other, self
        )

    def __gt__(self, other_prior):
        """
        Add an assertion that values associated with this prior are greater.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            lower physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior_model.assertion import GreaterThanLessThanAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanAssertion(
            greater=unwrap(self),
            lower=unwrap(other_prior)
        )

    def __lt__(self, other_prior):
        """
        Add an assertion that values associated with this prior are lower.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            greater physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior_model.assertion import GreaterThanLessThanAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanAssertion(
            lower=unwrap(self),
            greater=unwrap(other_prior)
        )

    def __ge__(self, other_prior):
        """
        Add an assertion that values associated with this prior are greater or equal.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            lower physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior_model.assertion import GreaterThanLessThanEqualAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanEqualAssertion(
            greater=unwrap(self),
            lower=unwrap(other_prior)
        )

    def __le__(self, other_prior):
        """
        Add an assertion that values associated with this prior are lower or equal.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            greater physical values.

        Returns
        -------
        An assertion object
        """
        from autofit.mapper.prior_model.assertion import GreaterThanLessThanEqualAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanEqualAssertion(
            lower=unwrap(self),
            greater=unwrap(other_prior)
        )
