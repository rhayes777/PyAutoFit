from autofit import exc

raise exc.FitException(
    "The normalization (norm) supplied to the plotter is not a valid string (must be "
    "{linear, log, symmetric_log})"
)
