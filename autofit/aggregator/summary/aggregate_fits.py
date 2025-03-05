from enum import Enum

from astropy.io import fits

from autofit.aggregator import Aggregator
from autofit.aggregator.summary.subplot_filename import subplot_filename


class Fit(Enum):
    ModelImage = "MODEL_IMAGE"
    ResidualMap = "RESIDUAL_MAP"
    NormalizedResidualMap = "NORMALIZED_RESIDUAL_MAP"
    ChiSquaredMap = "CHI_SQUARED_MAP"


class AggregateFITS:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def extract_fits(self, *hdus):
        output = [fits.PrimaryHDU()]
        for result in self.aggregator:
            for hdu in hdus:
                source = result.value(subplot_filename(hdu))
                source_hdu = source[source.index_of(hdu.value)]
                output.append(
                    fits.ImageHDU(
                        data=source_hdu.data,
                        header=source_hdu.header,
                    )
                )

        return fits.HDUList(output)
