.. _model:

Model
=====

.. code-block:: python

    class Gaussian:
        def __init__(
            self,
            centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
            normalization=1.0,  # <- are the Gaussian``s model parameters.
            sigma=5.0,
        ):
            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma