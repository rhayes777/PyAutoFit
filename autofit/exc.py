class PriorException(Exception):
    pass


class MultiNestException(Exception):
    pass


class CheckpointException(Exception):
    """
    Thrown when a non linear optimiser recovery fails due to assertion failure
    """
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
