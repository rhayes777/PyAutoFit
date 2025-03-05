import re
from enum import Enum
from typing import List

from astropy.io import fits
from pathlib import Path

from astropy.io.fits.hdu.image import _ImageBaseHDU

from autofit.aggregator import Aggregator


def subplot_filename(subplot: Enum) -> str:
    subplot_type = subplot.__class__
    return (
        re.sub(
            r"([A-Z])",
            r"_\1",
            subplot_type.__name__.replace("FITS", ""),
        )
        .lower()
        .lstrip("_")
    )


class FitFITS(Enum):
    ModelImage = "MODEL_IMAGE"
    ResidualMap = "RESIDUAL_MAP"
    NormalizedResidualMap = "NORMALIZED_RESIDUAL_MAP"
    ChiSquaredMap = "CHI_SQUARED_MAP"


class AggregateFITS:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def _hdus(self, result, *hdus) -> List[fits.ImageHDU]:
        row = []
        for hdu in hdus:
            source = result.value(subplot_filename(hdu))
            source_hdu = source[source.index_of(hdu.value)]
            row.append(
                fits.ImageHDU(
                    data=source_hdu.data,
                    header=source_hdu.header,
                )
            )
        return row

    def extract_fits(self, *hdus: Enum):
        output = [fits.PrimaryHDU()]
        for result in self.aggregator:
            output.extend(self._hdus(result, *hdus))

        return fits.HDUList(output)

    def output_to_folder(
        self,
        folder: Path,
        *hdus: Enum,
        name: str = "name",
    ):
        folder.mkdir(parents=True, exist_ok=True)

        for result in self.aggregator:
            name = f"{getattr(result, name)}.fits"
            fits.HDUList(
                [fits.PrimaryHDU()]
                + self._hdus(
                    result,
                    *hdus,
                )
            ).writeto(folder / name)
