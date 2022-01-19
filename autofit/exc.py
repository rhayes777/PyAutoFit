from autoconf.exc import PriorException


class MessageException(PriorException):
    """
    Raised when some assertion about the parameterization of a message is not met
    """


class PathsException(Exception):
    pass


class FitException(Exception):
    """
    An exception to be thrown if the non linear search must resample; equivalent to returning an infinitely bad fit
    """

    pass


class PriorLimitException(FitException, PriorException):
    pass


class PipelineException(Exception):
    pass


class DeferredInstanceException(Exception):
    """
    Exception raised when an attempt is made to access an attribute or function of a
    deferred instance prior to instantiation
    """

    pass


class AggregatorException(Exception):
    pass


class GridSearchException(Exception):
    pass


class HistoryException(Exception):
    """
    Thrown when insufficient factor history is present for a given operation
    """

class SamplesException(Exception):
    pass


class SamplesWarning(Warning):
    pass