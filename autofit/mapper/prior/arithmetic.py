class ArithmeticMixin:
    def __add__(self, other):
        """
        Add this object to another object. Addition occurs
        after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be summed after
        realisation
        """
        from autofit.mapper.prior.compound import SumPrior
        return SumPrior(
            self, other
        )

    def __radd__(self, other):
        """
        Add this object to another object. Addition occurs
        after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be summed after
        realisation
        """
        from autofit.mapper.prior.compound import SumPrior
        return SumPrior(
            other, self
        )

    def __sub__(self, other):
        """
        Subtract another object from this object. Subtraction
        occurs after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be summed after
        realisation
        """
        return self + (-other)

    def __neg__(self):
        """
        Create an object representing the negation of this
        object.
        """
        from autofit.mapper.prior.compound import NegativePrior
        return NegativePrior(
            self
        )

    def __mul__(self, other):
        """
        Multiple another object by this object. Multiplication
        occurs after priors have been converted into values.

        Parameters
        ----------
        other

        Returns
        -------
        An object comprising two objects to be multiplied after
        realisation
        """
        from autofit.mapper.prior.compound import MultiplePrior
        return MultiplePrior(
            self, other
        )

    def __rmul__(self, other):
        from autofit.mapper.prior.compound import MultiplePrior
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
        from autofit.mapper.prior.assertion import GreaterThanLessThanAssertion, unwrap
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
        from autofit.mapper.prior.assertion import GreaterThanLessThanAssertion, unwrap
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
        from autofit.mapper.prior.assertion import GreaterThanLessThanEqualAssertion, unwrap
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
        from autofit.mapper.prior.assertion import GreaterThanLessThanEqualAssertion, unwrap
        # noinspection PyTypeChecker
        return GreaterThanLessThanEqualAssertion(
            lower=unwrap(self),
            greater=unwrap(other_prior)
        )
